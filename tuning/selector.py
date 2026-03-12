from typing import Dict, Any, List


def _normalize_profile_keys(profiles: Dict[Any, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """
    JSON-loaded benchmark dumps may contain batch keys as strings.
    Normalize everything to int so sorting and lookup are stable.
    """
    out: Dict[int, Dict[str, float]] = {}
    for k, v in profiles.items():
        out[int(k)] = v
    return out


def calculate_neighbor_penalty(candidate_profiles: Dict[Any, Dict[str, float]], target_m: int) -> float:
    """
    Penalizes candidates that perform terribly in neighboring batch regions.
    A candidate that is extremely fast at M=64 but 5x slower at M=16 or M=256
    is unstable and likely hitting some catastrophic occupancy cliff.
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

    penalty = 0.0
    target_metric = candidate_profiles[target_m]["local_ms"]

    for n_m in neighbors:
        n_metric = candidate_profiles[n_m]["local_ms"]

        if n_metric == float("inf") or target_metric == float("inf"):
            return float("inf")

        ratio = (n_metric / n_m) / (target_metric / target_m)

        if ratio > 5.0 or ratio < 0.2:
            penalty += target_metric * 0.5  # heavy instability penalty

    return penalty


class CalibrationSelectionError(Exception):
    """Raised when kernel configuration selection yields no viable candidates."""
    pass


def fallback_select_best_candidate(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Deterministic fallback logic.
    Selects the absolute best candidate for each evaluated bucket based purely
    on the lowest local_ms (concurrency=1).
    """
    if not candidate_results:
        return {}

    # Normalize profile keys
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
        best_score = float("inf")
        best_candidate = None

        for result in normalized_results:
            profiles = result["profiles"]

            if m not in profiles:
                continue

            local_ms = profiles[m]["local_ms"]

            if local_ms < best_score:
                best_score = local_ms
                best_candidate = result["candidate"]

        if best_candidate is not None:
            winners[m] = best_candidate

    return winners


def score_and_select_winners(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Selects the winning kernel configuration for each evaluated batch bucket.

    candidate_results format:
    [
        {
            "candidate": {"BLOCK_SIZE_M": 64, ...},
            "profiles": {
                16: {"local_ms": 1.2, "parallel_ms": 1.4},
                64: {"local_ms": 2.1, "parallel_ms": 2.5},
            }
        }, ...
    ]

    Returns mapping from M -> winning candidate dict.
    """
    if not candidate_results:
        return {}

    # Normalize profile keys
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
        best_score = float("inf")
        best_candidate = None

        for result in normalized_results:
            profiles = result["profiles"]

            if m not in profiles:
                continue

            local_ms = profiles[m]["local_ms"]
            parallel_ms = profiles[m].get("parallel_ms", float("inf"))

            if local_ms == float("inf") or parallel_ms == float("inf"):
                continue

            # Parallel pressure weighted heavier
            combined_score = (local_ms * 0.3) + (parallel_ms * 0.7)

            stability_penalty = calculate_neighbor_penalty(profiles, m)
            combined_score += stability_penalty

            if combined_score < best_score:
                best_score = combined_score
                best_candidate = result["candidate"]

        if best_candidate is not None:
            winners[m] = best_candidate

    return winners