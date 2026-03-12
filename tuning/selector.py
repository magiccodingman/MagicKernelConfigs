from typing import Dict, Any, List, Optional, Tuple
import math


# -------------------------------------------------------------------
# Selection tuning knobs
# -------------------------------------------------------------------

# Positive signals (bigger is better)
BONUS_WEIGHTS = {
    "single_tps": 0.10,
    "moderate_tps": 0.16,
    "heavy_tps": 0.24,
    "weighted_total_tps": 0.12,
    "curve_score": 0.22,
    "saturation_stability_pct": 0.08,
    "local_ms": 0.05,      # lower is better; handled separately
    "parallel_ms": 0.03,   # lower is better; handled separately
}

# Negative signals (bigger is worse)
PENALTY_WEIGHTS = {
    "entry_cliff_pct": 0.16,
    "two_to_four_cliff_pct": 0.10,
    "neighbor_instability_pct": 0.10,
    "curve_jitter_pct": 0.04,
}

# When a metric is missing, give it a neutral score so legacy benchmark dumps
# do not get hard-punished just for being older / less expressive.
MISSING_METRIC_NEUTRAL_SCORE = 0.5


class CalibrationSelectionError(Exception):
    """Raised when kernel configuration selection yields no viable candidates."""
    pass


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------

def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def _inverse_ms_to_tps(ms: float) -> float:
    if not _is_finite_number(ms) or ms <= 0:
        return 0.0
    return 1000.0 / ms


def _normalize_profile_keys(profiles: Dict[Any, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    JSON-loaded benchmark dumps may contain batch keys as strings.
    Normalize everything to int so sorting and lookup are stable.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in profiles.items():
        out[int(k)] = v
    return out


# -------------------------------------------------------------------
# Curve extraction / derived metrics
# -------------------------------------------------------------------

def _extract_curve_points(profile: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    curve = profile.get("concurrency_curve")
    if not isinstance(curve, dict):
        return []

    points: List[Tuple[int, Dict[str, Any]]] = []
    for k, v in curve.items():
        try:
            c = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        points.append((c, v))

    points.sort(key=lambda x: x[0])
    return points


def _derive_entry_cliff_pct(profile: Dict[str, Any]) -> Optional[float]:
    if _is_finite_number(profile.get("entry_cliff_pct")):
        return float(profile["entry_cliff_pct"])

    points = _extract_curve_points(profile)
    if not points:
        return None

    point_map = {c: p for c, p in points}
    if 1 not in point_map or 2 not in point_map:
        return None

    tps1 = _safe_float(point_map[1].get("total_tps"), 0.0)
    tps2 = _safe_float(point_map[2].get("total_tps"), 0.0)

    if tps1 <= 0:
        return None

    tp_meta = profile.get("tp_meta", {}) or {}
    group_count = max(1, int(tp_meta.get("group_count", 1)))
    ideal_groups_used = min(2, group_count)
    ideal_tps2 = tps1 * ideal_groups_used

    if ideal_tps2 <= 0:
        return None

    return max(0.0, 100.0 - ((tps2 / ideal_tps2) * 100.0))


def _derive_two_to_four_cliff_pct(profile: Dict[str, Any]) -> Optional[float]:
    if _is_finite_number(profile.get("two_to_four_cliff_pct")):
        return float(profile["two_to_four_cliff_pct"])

    points = _extract_curve_points(profile)
    if not points:
        return None

    point_map = {c: p for c, p in points}
    if 2 not in point_map or 4 not in point_map:
        return None

    tps2 = _safe_float(point_map[2].get("total_tps"), 0.0)
    tps4 = _safe_float(point_map[4].get("total_tps"), 0.0)

    tp_meta = profile.get("tp_meta", {}) or {}
    group_count = max(1, int(tp_meta.get("group_count", 1)))

    tps1 = _safe_float(point_map.get(1, {}).get("total_tps"), 0.0)
    if tps1 <= 0:
        return None

    ideal2 = tps1 * min(2, group_count)
    ideal4 = tps1 * min(4, group_count)

    actual_gain = max(0.0, tps4 - tps2)
    ideal_gain = max(1e-9, ideal4 - ideal2)

    return max(0.0, (1.0 - (actual_gain / ideal_gain)) * 100.0)


def _derive_saturation_stability_pct(profile: Dict[str, Any]) -> Optional[float]:
    if _is_finite_number(profile.get("saturation_stability_pct")):
        return float(profile["saturation_stability_pct"])

    points = _extract_curve_points(profile)
    if not points:
        return None

    heavy = []
    for c, p in points:
        if c >= 6:
            tps = _safe_float(p.get("total_tps"), 0.0)
            if tps > 0:
                heavy.append((c, tps))

    if not heavy:
        return None

    peak = max(tps for _, tps in heavy)
    last = heavy[-1][1]

    if peak <= 0:
        return None

    return (last / peak) * 100.0


def _calculate_curve_jitter_pct(profile: Dict[str, Any]) -> Optional[float]:
    """
    Penalizes ugly throughput-curve behavior:
    - excessive oscillation
    - regression at higher concurrency
    """
    points = _extract_curve_points(profile)
    if len(points) < 3:
        return None

    tps_values: List[float] = []
    for _, p in points:
        tps = _safe_float(p.get("total_tps"), 0.0)
        if tps > 0:
            tps_values.append(tps)

    if len(tps_values) < 3:
        return None

    absolute_moves = []
    regressions = []

    for i in range(1, len(tps_values)):
        prev = max(tps_values[i - 1], 1e-9)
        curr = tps_values[i]

        absolute_moves.append(abs(curr - prev) / prev * 100.0)

        if curr < prev:
            regressions.append((prev - curr) / prev * 100.0)

    if not absolute_moves:
        return None

    mean_move = sum(absolute_moves) / len(absolute_moves)
    mean_regression = (sum(regressions) / len(regressions)) if regressions else 0.0

    # Regressions are more harmful than ordinary curvature.
    jitter = (mean_move * 0.30) + (mean_regression * 0.70)
    return min(100.0, jitter)


# -------------------------------------------------------------------
# Profile metric extraction
# -------------------------------------------------------------------

def _extract_profile_metrics(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts a serving-aware metric bundle from either:
    - new benchmarker output, or
    - older legacy {local_ms, parallel_ms} output
    """
    local_ms = _safe_float(profile.get("local_ms"), float("inf"))
    parallel_ms = _safe_float(profile.get("parallel_ms"), float("inf"))

    single_tps = profile.get("single_tps")
    if not _is_finite_number(single_tps):
        single_tps = _inverse_ms_to_tps(local_ms)

    moderate_tps = profile.get("moderate_tps")
    if not _is_finite_number(moderate_tps):
        moderate_tps = _inverse_ms_to_tps(parallel_ms)
        if moderate_tps <= 0:
            moderate_tps = single_tps

    heavy_tps = profile.get("heavy_tps")
    if not _is_finite_number(heavy_tps):
        heavy_tps = moderate_tps

    weighted_total_tps = profile.get("weighted_total_tps")
    if not _is_finite_number(weighted_total_tps):
        weighted_total_tps = (
            (float(single_tps) * 0.25) +
            (float(moderate_tps) * 0.35) +
            (float(heavy_tps) * 0.40)
        )

    curve_score = profile.get("curve_score")
    if not _is_finite_number(curve_score):
        # Legacy bridge:
        # If we have only latency-style fields, this becomes a throughput-ish proxy.
        curve_score = weighted_total_tps
        if curve_score <= 0 and _is_finite_number(parallel_ms) and parallel_ms > 0:
            curve_score = 1000.0 / parallel_ms

    entry_cliff_pct = _derive_entry_cliff_pct(profile)
    two_to_four_cliff_pct = _derive_two_to_four_cliff_pct(profile)
    saturation_stability_pct = _derive_saturation_stability_pct(profile)
    curve_jitter_pct = _calculate_curve_jitter_pct(profile)

    return {
        "local_ms": local_ms,
        "parallel_ms": parallel_ms,
        "single_tps": float(single_tps),
        "moderate_tps": float(moderate_tps),
        "heavy_tps": float(heavy_tps),
        "weighted_total_tps": float(weighted_total_tps),
        "curve_score": float(curve_score),
        "entry_cliff_pct": float(entry_cliff_pct) if entry_cliff_pct is not None else float("nan"),
        "two_to_four_cliff_pct": float(two_to_four_cliff_pct) if two_to_four_cliff_pct is not None else float("nan"),
        "saturation_stability_pct": float(saturation_stability_pct) if saturation_stability_pct is not None else float("nan"),
        "curve_jitter_pct": float(curve_jitter_pct) if curve_jitter_pct is not None else float("nan"),
    }


def _profile_quality_scalar(profile: Dict[str, Any]) -> float:
    """
    Used for neighboring bucket stability checks.
    Bigger is better.
    """
    m = _extract_profile_metrics(profile)

    quality = (
        (m["curve_score"] * 0.40) +
        (m["weighted_total_tps"] * 0.25) +
        (m["heavy_tps"] * 0.20) +
        (m["single_tps"] * 0.15)
    )

    if quality > 0:
        return quality

    if _is_finite_number(m["parallel_ms"]) and m["parallel_ms"] > 0:
        return 1000.0 / m["parallel_ms"]

    if _is_finite_number(m["local_ms"]) and m["local_ms"] > 0:
        return 1000.0 / m["local_ms"]

    return 0.0


# -------------------------------------------------------------------
# Neighbor penalty
# -------------------------------------------------------------------

def calculate_neighbor_penalty(candidate_profiles: Dict[Any, Dict[str, Any]], target_m: int) -> float:
    """
    Penalizes candidates that are suspiciously unstable across neighboring M buckets.

    This is no longer purely local_ms-based. It now compares a serving-aware
    quality scalar so a candidate can be punished if it looks amazing at one
    bucket but falls apart in adjacent regions.

    Returns a penalty percentage in [0, 100].
    """
    candidate_profiles = _normalize_profile_keys(candidate_profiles)

    sorted_ms = sorted(candidate_profiles.keys())
    if target_m not in sorted_ms:
        return 0.0

    idx = sorted_ms.index(target_m)
    neighbors = []

    if idx > 0:
        neighbors.append(sorted_ms[idx - 1])
    if idx < len(sorted_ms) - 1:
        neighbors.append(sorted_ms[idx + 1])

    if not neighbors:
        return 0.0

    target_quality = _profile_quality_scalar(candidate_profiles[target_m])
    if target_quality <= 0 or not math.isfinite(target_quality):
        return 100.0

    penalties = []

    for n_m in neighbors:
        neighbor_quality = _profile_quality_scalar(candidate_profiles[n_m])

        if neighbor_quality <= 0 or not math.isfinite(neighbor_quality):
            penalties.append(100.0)
            continue

        ratio = neighbor_quality / target_quality

        # Too much worse nearby = likely a cliff.
        if ratio < 0.55:
            penalties.append(min(100.0, ((0.55 - ratio) / 0.55) * 100.0))

        # Too much better nearby can also imply this bucket is a weird local failure pocket.
        elif ratio > 1.85:
            penalties.append(min(100.0, ((ratio - 1.85) / 1.85) * 60.0))

    if not penalties:
        return 0.0

    return min(100.0, sum(penalties))


# -------------------------------------------------------------------
# Candidate viability and normalization
# -------------------------------------------------------------------

def _is_viable_profile_metrics(metrics: Dict[str, float]) -> bool:
    if _is_finite_number(metrics.get("local_ms")) and metrics["local_ms"] > 0:
        return True

    if metrics.get("curve_score", 0.0) > 0:
        return True

    if metrics.get("single_tps", 0.0) > 0:
        return True

    return False


def _normalize_metric_across_rows(
    rows: List[Dict[str, Any]],
    metric_name: str,
    larger_is_better: bool,
    missing_score: float = MISSING_METRIC_NEUTRAL_SCORE,
) -> Dict[int, float]:
    """
    Min-max normalization for one metric across a candidate set for a single M bucket.

    Returns scores in [0, 1], where 1 is best.
    Missing values get a neutral score by default.
    """
    present_values = []

    for row in rows:
        raw = row["metrics"].get(metric_name)
        if _is_finite_number(raw):
            present_values.append(float(raw))

    if not present_values:
        return {i: missing_score for i in range(len(rows))}

    v_min = min(present_values)
    v_max = max(present_values)

    if math.isclose(v_min, v_max, rel_tol=1e-12, abs_tol=1e-12):
        return {i: 1.0 for i in range(len(rows))}

    out: Dict[int, float] = {}

    for i, row in enumerate(rows):
        raw = row["metrics"].get(metric_name)

        if not _is_finite_number(raw):
            out[i] = missing_score
            continue

        raw = float(raw)

        if larger_is_better:
            out[i] = (raw - v_min) / (v_max - v_min)
        else:
            out[i] = (v_max - raw) / (v_max - v_min)

    return out


def _build_bucket_rows(candidate_results: List[Dict[str, Any]], m: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for result in candidate_results:
        profiles = result["profiles"]
        if m not in profiles:
            continue

        profile = profiles[m]
        metrics = _extract_profile_metrics(profile)
        metrics["neighbor_instability_pct"] = calculate_neighbor_penalty(profiles, m)

        if not _is_viable_profile_metrics(metrics):
            continue

        rows.append({
            "candidate": result["candidate"],
            "profiles": profiles,
            "profile": profile,
            "metrics": metrics,
        })

    return rows


# -------------------------------------------------------------------
# Fallback selection
# -------------------------------------------------------------------

def _fallback_sort_key(profile: Dict[str, Any]) -> Tuple[float, ...]:
    """
    Deterministic fallback ordering.
    Lower tuple wins.
    """
    m = _extract_profile_metrics(profile)

    # Use negatives for "higher is better" values so min(tuple) still works.
    return (
        -m.get("curve_score", 0.0),
        -m.get("weighted_total_tps", 0.0),
        -m.get("heavy_tps", 0.0),
        -m.get("moderate_tps", 0.0),
        -m.get("single_tps", 0.0),
        m.get("entry_cliff_pct", float("inf")) if _is_finite_number(m.get("entry_cliff_pct")) else 50.0,
        m.get("two_to_four_cliff_pct", float("inf")) if _is_finite_number(m.get("two_to_four_cliff_pct")) else 50.0,
        -(m.get("saturation_stability_pct", 0.0) if _is_finite_number(m.get("saturation_stability_pct")) else 50.0),
        m.get("parallel_ms", float("inf")),
        m.get("local_ms", float("inf")),
    )


def fallback_select_best_candidate(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Deterministic fallback logic.

    Unlike the old fallback, this no longer chooses purely by lowest local_ms.
    It prefers curve-aware metrics when available, then falls back cleanly.
    """
    if not candidate_results:
        return {}

    normalized_results = []
    for result in candidate_results:
        normalized_results.append({
            "candidate": result["candidate"],
            "profiles": _normalize_profile_keys(result["profiles"]),
        })

    eval_buckets = sorted({
        bucket
        for result in normalized_results
        for bucket in result["profiles"].keys()
    })

    winners: Dict[int, Dict[str, Any]] = {}

    for m in eval_buckets:
        best_candidate = None
        best_key = None

        for result in normalized_results:
            profiles = result["profiles"]
            if m not in profiles:
                continue

            profile = profiles[m]
            metrics = _extract_profile_metrics(profile)
            if not _is_viable_profile_metrics(metrics):
                continue

            current_key = _fallback_sort_key(profile)

            if best_key is None or current_key < best_key:
                best_key = current_key
                best_candidate = result["candidate"]

        if best_candidate is not None:
            winners[m] = best_candidate

    return winners


# -------------------------------------------------------------------
# Main scoring selector
# -------------------------------------------------------------------

def score_and_select_winners(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Serving-aware selector.

    Prefers candidates that:
    - keep single-stream strong
    - sustain throughput at moderate/heavy concurrency
    - avoid ugly 1→2 / 2→4 cliffs
    - remain stable across neighboring M buckets

    Still works with legacy benchmark dumps that only contain:
    - local_ms
    - parallel_ms
    """
    if not candidate_results:
        return {}

    normalized_results = []
    for result in candidate_results:
        normalized_results.append({
            "candidate": result["candidate"],
            "profiles": _normalize_profile_keys(result["profiles"]),
        })

    eval_buckets = sorted({
        bucket
        for result in normalized_results
        for bucket in result["profiles"].keys()
    })

    winners: Dict[int, Dict[str, Any]] = {}

    for m in eval_buckets:
        rows = _build_bucket_rows(normalized_results, m)
        if not rows:
            continue

        bonus_scores: Dict[str, Dict[int, float]] = {
            "single_tps": _normalize_metric_across_rows(rows, "single_tps", larger_is_better=True),
            "moderate_tps": _normalize_metric_across_rows(rows, "moderate_tps", larger_is_better=True),
            "heavy_tps": _normalize_metric_across_rows(rows, "heavy_tps", larger_is_better=True),
            "weighted_total_tps": _normalize_metric_across_rows(rows, "weighted_total_tps", larger_is_better=True),
            "curve_score": _normalize_metric_across_rows(rows, "curve_score", larger_is_better=True),
            "saturation_stability_pct": _normalize_metric_across_rows(rows, "saturation_stability_pct", larger_is_better=True),
            "local_ms": _normalize_metric_across_rows(rows, "local_ms", larger_is_better=False),
            "parallel_ms": _normalize_metric_across_rows(rows, "parallel_ms", larger_is_better=False),
        }

        penalty_scores: Dict[str, Dict[int, float]] = {
            "entry_cliff_pct": _normalize_metric_across_rows(rows, "entry_cliff_pct", larger_is_better=False),
            "two_to_four_cliff_pct": _normalize_metric_across_rows(rows, "two_to_four_cliff_pct", larger_is_better=False),
            "neighbor_instability_pct": _normalize_metric_across_rows(rows, "neighbor_instability_pct", larger_is_better=False),
            "curve_jitter_pct": _normalize_metric_across_rows(rows, "curve_jitter_pct", larger_is_better=False),
        }

        best_idx = None
        best_score = -float("inf")

        for i, row in enumerate(rows):
            score = 0.0

            # Positive components
            for metric_name, weight in BONUS_WEIGHTS.items():
                score += bonus_scores[metric_name][i] * weight

            # Negative components
            # penalty_scores are normalized with "lower is better", so higher score is better.
            # Convert that to a penalty by subtracting the *badness* instead of the goodness.
            for metric_name, weight in PENALTY_WEIGHTS.items():
                goodness = penalty_scores[metric_name][i]
                badness = 1.0 - goodness
                score -= badness * weight

            # Small deterministic tie-break bias favoring stronger raw curve score
            raw_curve_score = row["metrics"].get("curve_score", 0.0)
            if _is_finite_number(raw_curve_score):
                score += float(raw_curve_score) * 1e-9

            # Tiny bias for lower local latency if still tied
            local_ms = row["metrics"].get("local_ms", float("inf"))
            if _is_finite_number(local_ms) and local_ms > 0:
                score += (1.0 / local_ms) * 1e-9

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            winners[m] = rows[best_idx]["candidate"]

    return winners