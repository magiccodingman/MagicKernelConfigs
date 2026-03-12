import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any, List
import os
import json
import math
import statistics
import concurrent.futures

from tuning.kernel_harness import get_harness_code

# -------------------------------------------------------------------
# Default serving-shaped benchmark profile
# -------------------------------------------------------------------

DEFAULT_DECODE_BUCKETS = [1, 2, 4, 8, 16]
DEFAULT_PREFILL_BUCKETS = [64, 256]
DEFAULT_CONCURRENCY_POINTS = [1, 2, 4, 6, 8, 10]

DEFAULT_OPS_PER_REQUEST = 8
DEFAULT_WARMUP_ITERS = 3
DEFAULT_TIMEOUT_SECS = 45.0


# -------------------------------------------------------------------
# Low-level isolated worker benchmark
# -------------------------------------------------------------------

def _invalid_worker_payload(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "avg_ms": float("inf"),
        "total_ms": float("inf"),
        "timed_iters": 0,
        "actual_backend": "error",
        "backend_proof": reason,
    }


def run_isolated_benchmark_on_gpu(
    candidate: Dict[str, Any],
    m: int,
    n: int,
    k: int,
    gpu_id: str,
    backend: str,
    dtype_family: str,
    is_moe: bool,
    timed_iters: int,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """
    Runs the backend kernel in an isolated subprocess pinned to one GPU.

    Returns:
        {
            "ok": bool,
            "avg_ms": float,
            "total_ms": float,
            "timed_iters": int,
            "actual_backend": str,
            "backend_proof": str,
            ...
        }
    """
    timed_iters = max(1, int(timed_iters))
    warmup_iters = max(0, int(warmup_iters))

    harness_code = get_harness_code(backend, dtype_family, is_moe) + textwrap.dedent(f"""\
        import json
        import sys
        import time
        import torch

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if "{backend}" == "aiter_triton" and "{dtype_family}" == "fp8":
                # Keep float16 masters; AITER adapter quantizes internally.
                a = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
                b = torch.randn(({k}, {n}), device=device, dtype=torch.float16)

            elif "{dtype_family}" == "fp8":
                a = torch.randn(({m}, {k}), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
                b = torch.randn(({k}, {n}), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)

            elif "{dtype_family}" == "int8":
                a = torch.randint(-128, 127, ({m}, {k}), device=device, dtype=torch.int8)
                b = torch.randint(-128, 127, ({k}, {n}), device=device, dtype=torch.int8)

            else:
                # FP4 / INT4 path placeholders still fall back here unless your harness
                # internally redirects them through backend-specific adapters.
                a = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
                b = torch.randn(({k}, {n}), device=device, dtype=torch.float16)

            candidate_params = {candidate}

            warmup_iters = {warmup_iters}
            timed_iters = {timed_iters}

            for _ in range(warmup_iters):
                run_backend_kernel(a, b, candidate_params, "{backend}")

            if device != "cpu":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(timed_iters):
                run_backend_kernel(a, b, candidate_params, "{backend}")
            if device != "cpu":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_ms = (end - start) * 1000.0
            avg_ms = total_ms / max(timed_iters, 1)

            print(json.dumps({{
                "ok": True,
                "avg_ms": avg_ms,
                "total_ms": total_ms,
                "timed_iters": timed_iters,
                "actual_backend": ACTUAL_BACKEND,
                "backend_proof": BACKEND_PROOF,
                "requested_backend": "{backend}"
            }}))
            sys.exit(0)

        except Exception as e:
            print(json.dumps({{
                "ok": False,
                "avg_ms": None,
                "total_ms": None,
                "timed_iters": 0,
                "actual_backend": "error",
                "backend_proof": str(e),
                "requested_backend": "{backend}"
            }}), file=sys.stderr)
            sys.exit(1)
    """)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness_code)
        temp_path = Path(f.name)

    try:
        env = os.environ.copy()

        project_root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH", "")

        paths = [project_root]
        if existing:
            paths.append(existing)

        env["PYTHONPATH"] = ":".join(paths)

        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        if backend == "aiter_triton":
            env["MAGIC_AITER_RUNNER"] = "tuning.aiter_runner:run_aiter_triton_kernel"

            if is_moe:
                dtype_map = {
                    "fp8": "aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale:moe_gemm_a8w8_blockscale",
                    "int8": "aiter.ops.triton.moe.moe_op_gemm_a8w8:moe_gemm_a8w8",
                    "fp4": "aiter.ops.triton.moe.moe_op_gemm_a8w4:moe_gemm_a8w4",
                    "int4": "aiter.ops.triton.moe.moe_op_gemm_a4w4:moe_gemm_a4w4",
                }
            else:
                dtype_map = {
                    "fp8": "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale:gemm_a8w8_blockscale",
                    "int8": "aiter.ops.triton.gemm.basic.gemm_a8w8:gemm_a8w8",
                    "fp4": "aiter.ops.triton.gemm.basic.gemm_afp4wfp4:gemm_afp4wfp4",
                    "int4": "aiter.ops.triton.gemm.basic.gemm_a8wfp4:gemm_a8wfp4",
                }

            callable_name = dtype_map.get(dtype_family)
            if callable_name is None:
                return _invalid_worker_payload(f"unsupported dtype_family for aiter_triton: {dtype_family}")

            env["MAGIC_AITER_CALLABLE"] = callable_name

            env["VLLM_ROCM_USE_AITER"] = "1"
            env["VLLM_ROCM_USE_AITER_LINEAR"] = "1"
            env["VLLM_ROCM_USE_AITER_TRITON_GEMM"] = "1"

            env["VLLM_ROCM_USE_AITER_MHA"] = "0"
            env["VLLM_ROCM_USE_AITER_MLA"] = "0"
            env["VLLM_ROCM_USE_AITER_RMSNORM"] = "0"
            env["VLLM_ROCM_USE_AITER_PAGED_ATTN"] = "0"
            env["VLLM_ROCM_USE_AITER_TRITON_ROPE"] = "0"

            env["VLLM_ROCM_USE_AITER_MOE"] = "1" if is_moe else "0"

        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env=env
        )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            return _invalid_worker_payload(stderr or "subprocess returned non-zero")

        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if not lines:
            return _invalid_worker_payload("no stdout payload")

        payload = json.loads(lines[-1])

        if payload.get("requested_backend") != backend:
            return _invalid_worker_payload("requested_backend mismatch")

        if payload.get("actual_backend") != backend:
            return _invalid_worker_payload("actual_backend mismatch")

        if backend == "aiter_triton":
            expected = env["MAGIC_AITER_CALLABLE"]
            if expected not in str(payload.get("backend_proof", "")):
                return _invalid_worker_payload("aiter callable proof mismatch")

        if payload.get("avg_ms") is None or payload.get("total_ms") is None:
            return _invalid_worker_payload("avg_ms/total_ms missing")

        payload["ok"] = True
        return payload

    except subprocess.TimeoutExpired:
        return _invalid_worker_payload("timeout")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return _invalid_worker_payload(f"parse_error: {e}")
    finally:
        temp_path.unlink(missing_ok=True)


# -------------------------------------------------------------------
# TP-aware grouping / scheduling helpers
# -------------------------------------------------------------------

def _visible_gpu_ids(available_gpus: int) -> List[str]:
    if available_gpus <= 0:
        return ["0"]
    return [str(i) for i in range(available_gpus)]


def _build_tp_groups(available_gpus: int, tp_target: int) -> Dict[str, Any]:
    """
    Builds GPU groups used to emulate tensor-parallel serving.

    Behavior:
    - TP target <= available GPUs: use that TP width and form disjoint groups.
    - TP target > available GPUs: best-effort; use all GPUs as one TP group.
    """
    gpu_ids = _visible_gpu_ids(available_gpus)
    detected_gpu_count = len(gpu_ids)

    effective_tp = max(1, min(max(1, tp_target), detected_gpu_count))

    if detected_gpu_count >= effective_tp:
        full_group_count = detected_gpu_count // effective_tp
        groups = []
        for i in range(full_group_count):
            start = i * effective_tp
            groups.append(gpu_ids[start:start + effective_tp])

        if not groups:
            groups = [gpu_ids[:effective_tp]]
    else:
        groups = [gpu_ids]

    used_gpu_ids = [x for g in groups for x in g]
    leftover_gpu_ids = [x for x in gpu_ids if x not in used_gpu_ids]

    return {
        "requested_tp": int(tp_target),
        "effective_tp": int(effective_tp),
        "detected_gpu_count": int(detected_gpu_count),
        "group_count": int(len(groups)),
        "groups": groups,
        "leftover_gpu_ids": leftover_gpu_ids,
        "best_effort_only": bool(tp_target > detected_gpu_count),
    }


def _distribute_requests(concurrency: int, group_count: int) -> List[int]:
    """
    Evenly distribute 'concurrency' request units across active TP groups.
    """
    group_count = max(1, int(group_count))
    concurrency = max(1, int(concurrency))

    base = concurrency // group_count
    rem = concurrency % group_count

    out = []
    for i in range(group_count):
        out.append(base + (1 if i < rem else 0))
    return out


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(v)]
    if not vals:
        return float("inf")
    return float(statistics.mean(vals))


def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return fallback
    return float(numerator / denominator)


def _tp_group_wall_ms(worker_payloads: List[Dict[str, Any]]) -> float:
    """
    In TP, the slowest worker gates the group's step time.
    """
    vals = [float(x["total_ms"]) for x in worker_payloads if x.get("ok")]
    return max(vals) if vals else float("inf")


def _tp_group_avg_ms(worker_payloads: List[Dict[str, Any]]) -> float:
    """
    Per-op TP latency proxy. Again, slowest worker is the gate.
    """
    vals = [float(x["avg_ms"]) for x in worker_payloads if x.get("ok")]
    return max(vals) if vals else float("inf")


# -------------------------------------------------------------------
# Group/scenario execution
# -------------------------------------------------------------------

def _run_tp_group_benchmark(
    candidate: Dict[str, Any],
    m: int,
    n: int,
    k: int,
    gpu_group: List[str],
    backend: str,
    dtype_family: str,
    is_moe: bool,
    request_units: int,
    ops_per_request: int,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
) -> Dict[str, Any]:
    """
    Runs a best-effort TP-group benchmark. Every GPU in the TP group launches
    the same GEMM workload concurrently. Total group wall time is gated by the
    slowest worker.

    request_units:
        Number of "request units" queued onto this TP group for this scenario.

    ops_per_request:
        Number of timed kernel invocations used to represent one request unit.
        This smooths noise and turns throughput into a more stable signal.
    """
    request_units = max(1, int(request_units))
    timed_iters = max(1, int(request_units * ops_per_request))

    worker_payloads: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_group)) as executor:
        futures = []
        for gpu_id in gpu_group:
            futures.append(executor.submit(
                run_isolated_benchmark_on_gpu,
                candidate,
                m,
                n,
                k,
                gpu_id,
                backend,
                dtype_family,
                is_moe,
                timed_iters,
                warmup_iters,
                DEFAULT_TIMEOUT_SECS,
            ))

        for f in concurrent.futures.as_completed(futures):
            worker_payloads.append(f.result())

    if not worker_payloads or not all(x.get("ok") for x in worker_payloads):
        return {
            "ok": False,
            "gpu_group": gpu_group,
            "request_units": request_units,
            "timed_iters": timed_iters,
            "group_total_ms": float("inf"),
            "group_avg_ms": float("inf"),
            "workers": worker_payloads,
        }

    group_total_ms = _tp_group_wall_ms(worker_payloads)
    group_avg_ms = _tp_group_avg_ms(worker_payloads)

    return {
        "ok": True,
        "gpu_group": gpu_group,
        "request_units": request_units,
        "timed_iters": timed_iters,
        "group_total_ms": group_total_ms,
        "group_avg_ms": group_avg_ms,
        "workers": worker_payloads,
    }


def _run_concurrency_scenario(
    candidate: Dict[str, Any],
    m: int,
    n: int,
    k: int,
    backend: str,
    dtype_family: str,
    is_moe: bool,
    tp_meta: Dict[str, Any],
    concurrency: int,
    ops_per_request: int,
) -> Dict[str, Any]:
    """
    Runs one serving-shaped scenario.

    Example:
    - TP2 on 4 GPUs => 2 TP groups.
    - concurrency=6 => request units distribute as [3, 3] across the 2 groups.
    - Each group runs enough timed kernel ops to represent its assigned load.
    - The scenario wall time is the slowest TP group.
    """
    groups = tp_meta["groups"]
    total_groups = max(1, len(groups))
    active_groups = min(total_groups, max(1, int(concurrency)))

    active_gpu_groups = groups[:active_groups]
    request_distribution = _distribute_requests(concurrency, active_groups)

    group_payloads: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=active_groups) as executor:
        futures = []
        for gpu_group, request_units in zip(active_gpu_groups, request_distribution):
            futures.append(executor.submit(
                _run_tp_group_benchmark,
                candidate,
                m,
                n,
                k,
                gpu_group,
                backend,
                dtype_family,
                is_moe,
                request_units,
                ops_per_request,
                DEFAULT_WARMUP_ITERS,
            ))

        for f in concurrent.futures.as_completed(futures):
            group_payloads.append(f.result())

    if not group_payloads or not all(x.get("ok") for x in group_payloads):
        return {
            "ok": False,
            "concurrency": int(concurrency),
            "active_groups": int(active_groups),
            "request_distribution": request_distribution,
            "wall_ms": float("inf"),
            "total_tps": 0.0,
            "ms_per_request_proxy": float("inf"),
            "ms_per_op_proxy": float("inf"),
            "groups": group_payloads,
        }

    wall_ms = max(float(x["group_total_ms"]) for x in group_payloads)
    total_ops = concurrency * ops_per_request

    total_tps = 0.0
    if wall_ms > 0 and math.isfinite(wall_ms):
        total_tps = total_ops / (wall_ms / 1000.0)

    ms_per_request_proxy = wall_ms / max(concurrency, 1)
    ms_per_op_proxy = wall_ms / max(total_ops, 1)

    return {
        "ok": True,
        "concurrency": int(concurrency),
        "active_groups": int(active_groups),
        "request_distribution": request_distribution,
        "wall_ms": wall_ms,
        "total_ops": int(total_ops),
        "total_tps": total_tps,
        "ms_per_request_proxy": ms_per_request_proxy,
        "ms_per_op_proxy": ms_per_op_proxy,
        "groups": group_payloads,
    }


# -------------------------------------------------------------------
# Curve summary / serving metrics
# -------------------------------------------------------------------

def _collapse_pct(prev_val: float, next_val: float) -> float:
    """
    Percentage drop from prev -> next.
    Positive means worse.
    """
    if not math.isfinite(prev_val) or prev_val <= 0:
        return 100.0
    return max(0.0, (1.0 - (next_val / prev_val)) * 100.0)


def _summarize_curve(
    curve_by_concurrency: Dict[str, Dict[str, Any]],
    group_count: int,
) -> Dict[str, Any]:
    """
    Converts raw concurrency samples into serving-shaped summary metrics.
    """
    ordered_keys = sorted((int(k) for k in curve_by_concurrency.keys()))
    ordered = [curve_by_concurrency[str(k)] for k in ordered_keys]

    single = curve_by_concurrency.get("1")
    if not single or not single.get("ok"):
        return {
            "single_tps": 0.0,
            "moderate_tps": 0.0,
            "heavy_tps": 0.0,
            "entry_cliff_pct": 100.0,
            "two_to_four_cliff_pct": 100.0,
            "saturation_stability_pct": 0.0,
            "weighted_total_tps": 0.0,
            "curve_score": 0.0,
            "parallel_ms_proxy": float("inf"),
        }

    single_tps = float(single["total_tps"])

    # Scaling efficiency against ideal disjoint-group scaling
    for sample in ordered:
        c = int(sample["concurrency"])
        ideal_groups_used = min(c, max(1, group_count))
        ideal_total_tps = single_tps * ideal_groups_used
        efficiency_pct = _safe_ratio(sample["total_tps"], ideal_total_tps, 0.0) * 100.0
        sample["ideal_total_tps"] = ideal_total_tps
        sample["scaling_efficiency_pct"] = efficiency_pct

    moderate_keys = [k for k in ordered_keys if 2 <= k <= 4]
    heavy_keys = [k for k in ordered_keys if k >= 6]

    moderate_tps = _mean([curve_by_concurrency[str(k)]["total_tps"] for k in moderate_keys]) if moderate_keys else single_tps
    heavy_tps = _mean([curve_by_concurrency[str(k)]["total_tps"] for k in heavy_keys]) if heavy_keys else moderate_tps

    entry_cliff_pct = 0.0
    if "2" in curve_by_concurrency:
        entry_cliff_pct = max(0.0, 100.0 - curve_by_concurrency["2"].get("scaling_efficiency_pct", 0.0))

    two_to_four_cliff_pct = 0.0
    if "2" in curve_by_concurrency and "4" in curve_by_concurrency:
        tps2 = float(curve_by_concurrency["2"]["total_tps"])
        tps4 = float(curve_by_concurrency["4"]["total_tps"])
        ideal2 = float(curve_by_concurrency["2"]["ideal_total_tps"])
        ideal4 = float(curve_by_concurrency["4"]["ideal_total_tps"])

        actual_gain = max(0.0, tps4 - tps2)
        ideal_gain = max(1e-9, ideal4 - ideal2)

        two_to_four_cliff_pct = max(0.0, (1.0 - (actual_gain / ideal_gain)) * 100.0)

    saturation_stability_pct = 100.0
    if heavy_keys:
        heavy_tps_values = [curve_by_concurrency[str(k)]["total_tps"] for k in heavy_keys]
        heavy_peak = max(heavy_tps_values)
        heavy_last = curve_by_concurrency[str(max(heavy_keys))]["total_tps"]
        saturation_stability_pct = _safe_ratio(heavy_last, heavy_peak, 0.0) * 100.0

    # Weighted throughput objective
    weighted_total_tps = (
        (single_tps * 0.25) +
        (moderate_tps * 0.35) +
        (heavy_tps * 0.40)
    )

    # Penalties: punish ugly entry / cliff / unstable plateau
    penalty_factor = 1.0
    penalty_factor -= min(0.35, entry_cliff_pct / 250.0)
    penalty_factor -= min(0.25, two_to_four_cliff_pct / 400.0)
    penalty_factor -= min(0.20, max(0.0, 100.0 - saturation_stability_pct) / 300.0)
    penalty_factor = max(0.05, penalty_factor)

    curve_score = weighted_total_tps * penalty_factor

    parallel_ms_proxy = float("inf")
    if curve_score > 0:
        parallel_ms_proxy = 1000.0 / curve_score

    return {
        "single_tps": single_tps,
        "moderate_tps": moderate_tps,
        "heavy_tps": heavy_tps,
        "entry_cliff_pct": entry_cliff_pct,
        "two_to_four_cliff_pct": two_to_four_cliff_pct,
        "saturation_stability_pct": saturation_stability_pct,
        "weighted_total_tps": weighted_total_tps,
        "curve_score": curve_score,
        "parallel_ms_proxy": parallel_ms_proxy,
    }


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

class BenchmarkModes:
    @staticmethod
    def local_mode(
        candidate: Dict[str, Any],
        m: int,
        n: int,
        k: int,
        available_gpus: int,
        tp_target: int,
        backend: str,
        dtype_family: str,
        is_moe: bool,
    ) -> float:
        """
        Single-request TP-aware local measurement.
        This is no longer "one GPU in isolation" unless TP target == 1.
        """
        tp_meta = _build_tp_groups(available_gpus, tp_target)
        groups = tp_meta["groups"]
        if not groups:
            return float("inf")

        payload = _run_tp_group_benchmark(
            candidate=candidate,
            m=m,
            n=n,
            k=k,
            gpu_group=groups[0],
            backend=backend,
            dtype_family=dtype_family,
            is_moe=is_moe,
            request_units=1,
            ops_per_request=DEFAULT_OPS_PER_REQUEST,
            warmup_iters=DEFAULT_WARMUP_ITERS,
        )
        return float(payload["group_avg_ms"]) if payload.get("ok") else float("inf")

    @staticmethod
    def parallel_contention_mode(
        candidate: Dict[str, Any],
        m: int,
        n: int,
        k: int,
        available_gpus: int,
        tp_target: int,
        backend: str,
        dtype_family: str,
        is_moe: bool,
        concurrency: int = 2,
    ) -> float:
        """
        Throughput-aware parallel proxy.
        Returns a reciprocal-throughput ms-like score so older selectors can
        still minimize something meaningful.
        """
        tp_meta = _build_tp_groups(available_gpus, tp_target)
        scenario = _run_concurrency_scenario(
            candidate=candidate,
            m=m,
            n=n,
            k=k,
            backend=backend,
            dtype_family=dtype_family,
            is_moe=is_moe,
            tp_meta=tp_meta,
            concurrency=concurrency,
            ops_per_request=DEFAULT_OPS_PER_REQUEST,
        )
        if not scenario.get("ok"):
            return float("inf")

        total_tps = float(scenario["total_tps"])
        if total_tps <= 0:
            return float("inf")
        return 1000.0 / total_tps


def run_workload_profiles(
    candidate: Dict[str, Any],
    n: int,
    k: int,
    is_moe: bool,
    available_gpus: int,
    backend: str,
    dtype_family: str,
    tp_target: int = 1,
    max_concurrency: int = 10,
    ops_per_request: int = DEFAULT_OPS_PER_REQUEST,
) -> Dict[int, Dict[str, Any]]:
    """
    Benchmarks the candidate across decode/prefill M-buckets and a serving-shaped
    concurrency curve.

    Returns a dict keyed by M bucket. Each bucket includes:
      - legacy fields:
          local_ms
          parallel_ms
      - richer fields:
          single_tps
          moderate_tps
          heavy_tps
          entry_cliff_pct
          two_to_four_cliff_pct
          saturation_stability_pct
          curve_score
          concurrency_curve
          tp_meta
    """
    decode_buckets = list(DEFAULT_DECODE_BUCKETS)
    prefill_buckets = list(DEFAULT_PREFILL_BUCKETS)
    all_buckets = decode_buckets + prefill_buckets

    concurrency_points = [x for x in DEFAULT_CONCURRENCY_POINTS if x <= max_concurrency]
    if 1 not in concurrency_points:
        concurrency_points = [1] + concurrency_points
    concurrency_points = sorted(set(concurrency_points))

    tp_meta = _build_tp_groups(available_gpus, tp_target)
    results: Dict[int, Dict[str, Any]] = {}

    for m in all_buckets:
        curve_by_concurrency: Dict[str, Dict[str, Any]] = {}

        for concurrency in concurrency_points:
            scenario = _run_concurrency_scenario(
                candidate=candidate,
                m=m,
                n=n,
                k=k,
                backend=backend,
                dtype_family=dtype_family,
                is_moe=is_moe,
                tp_meta=tp_meta,
                concurrency=concurrency,
                ops_per_request=ops_per_request,
            )
            curve_by_concurrency[str(concurrency)] = scenario

        summary = _summarize_curve(curve_by_concurrency, tp_meta["group_count"])

        # local_ms: TP-aware single-request per-op latency proxy
        local_ms = float("inf")
        if curve_by_concurrency.get("1", {}).get("ok"):
            local_ms = float(curve_by_concurrency["1"]["ms_per_op_proxy"])

        # parallel_ms: reciprocal of weighted throughput curve score
        parallel_ms = float(summary["parallel_ms_proxy"])

        results[m] = {
            # backward-compatible keys
            "local_ms": local_ms,
            "parallel_ms": parallel_ms,

            # richer serving-aware metrics
            "phase": "decode" if m <= 16 else "prefill",
            "single_tps": summary["single_tps"],
            "moderate_tps": summary["moderate_tps"],
            "heavy_tps": summary["heavy_tps"],
            "entry_cliff_pct": summary["entry_cliff_pct"],
            "two_to_four_cliff_pct": summary["two_to_four_cliff_pct"],
            "saturation_stability_pct": summary["saturation_stability_pct"],
            "weighted_total_tps": summary["weighted_total_tps"],
            "curve_score": summary["curve_score"],

            # full raw curve for later selector upgrades
            "concurrency_curve": curve_by_concurrency,

            # runtime / grouping metadata
            "tp_meta": tp_meta,
            "ops_per_request": int(ops_per_request),
        }

    return results