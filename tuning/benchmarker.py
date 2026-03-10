import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any
import os
import json

from tuning.kernel_harness import get_harness_code

def run_isolated_benchmark_on_gpu(candidate: Dict[str, Any], m: int, n: int, k: int,
                                  gpu_id: str, backend: str, dtype_family: str, is_moe: bool) -> float:
    harness_code = get_harness_code(backend, dtype_family, is_moe) + textwrap.dedent(f"""\
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if "{dtype_family}" == "fp8":
                a = torch.randn(({m}, {k}), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
                b = torch.randn(({k}, {n}), device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
            elif "{dtype_family}" == "int8":
                a = torch.randint(-128, 127, ({m}, {k}), device=device, dtype=torch.int8)
                b = torch.randint(-128, 127, ({k}, {n}), device=device, dtype=torch.int8)
            else:
                a = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
                b = torch.randn(({k}, {n}), device=device, dtype=torch.float16)

            candidate_params = {candidate}

            for _ in range(3):
                run_backend_kernel(a, b, candidate_params, "{backend}")
            if device != "cpu":
                torch.cuda.synchronize()

            start = time.perf_counter()
            iters = 10
            for _ in range(iters):
                run_backend_kernel(a, b, candidate_params, "{backend}")
            if device != "cpu":
                torch.cuda.synchronize()
            end = time.perf_counter()

            avg_ms = ((end - start) / iters) * 1000.0
            print(json.dumps({{
                "avg_ms": avg_ms,
                "actual_backend": ACTUAL_BACKEND,
                "backend_proof": BACKEND_PROOF,
                "requested_backend": "{backend}"
            }}))
            sys.exit(0)

        except Exception as e:
            print(json.dumps({{
                "avg_ms": None,
                "actual_backend": "error",
                "backend_proof": str(e),
                "requested_backend": "{backend}"
            }}), file=sys.stderr)
            sys.exit(1)
    """)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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

            # Select correct AITER GEMM callable based on dtype
            if is_moe:
                dtype_map = {
                    "fp8": "moe_gemm_a8w8_blockscale",
                    "int8": "moe_gemm_a8w8",
                    "fp4": "moe_gemm_a8w4",
                    "int4": "moe_gemm_a4w4",
                }
            else:
                dtype_map = {
                    "fp8": "gemm_a8w8_blockscale",
                    "int8": "gemm_a8w8",
                    "fp4": "gemm_afp4wfp4",
                    "int4": "gemm_a8wfp4",
                }

            callable_name = dtype_map.get(dtype_family)
            if callable_name is None:
                return float("inf")

            env["MAGIC_AITER_CALLABLE"] = callable_name

            # Enable AITER Triton GEMM
            env["VLLM_ROCM_USE_AITER"] = "1"
            env["VLLM_ROCM_USE_AITER_LINEAR"] = "1"
            env["VLLM_ROCM_USE_AITER_TRITON_GEMM"] = "1"

            # Disable unrelated subsystems to isolate GEMM
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
            timeout=15,
            env=env
        )

        if result.returncode != 0:
            return float('inf')

        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if not lines:
            return float('inf')

        payload = json.loads(lines[-1])

        if payload["requested_backend"] != backend:
            return float('inf')

        if payload["actual_backend"] != backend:
            return float('inf')
        
        if backend == "aiter_triton":
            expected = env["MAGIC_AITER_CALLABLE"]
            if expected not in payload["backend_proof"]:
                return float("inf")

        if payload["avg_ms"] is None:
            return float('inf')

        return float(payload["avg_ms"])

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        return float('inf')
    finally:
        temp_path.unlink(missing_ok=True)

class BenchmarkModes:
    @staticmethod
    def local_mode(candidate: Dict[str, Any], m: int, n: int, k: int, backend: str, dtype_family: str, is_moe: bool) -> float:
        """Evaluates candidate kernel in isolation on a single GPU."""
        return run_isolated_benchmark_on_gpu(candidate, m, n, k, "0", backend, dtype_family, is_moe)

    @staticmethod
    def parallel_contention_mode(candidate: Dict[str, Any], m: int, n: int, k: int, gpus: int, backend: str, dtype_family: str, is_moe: bool) -> float:
        """
        Executes candidate kernels concurrently across available GPUs.
        Simulates tensor-parallel contention.
        """
        import concurrent.futures
        
        if gpus <= 1:
            return BenchmarkModes.local_mode(candidate, m, n, k, backend, dtype_family, is_moe)
            
        times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpus) as executor:
            futures = []
            for i in range(gpus):
                # We assume available GPUs are uniquely indexed starting from 0,
                # as the orchestrator isolated the system to only identical GPUs.
                futures.append(executor.submit(run_isolated_benchmark_on_gpu, candidate, m, n, k, str(i), backend, dtype_family, is_moe))
                
            for f in concurrent.futures.as_completed(futures):
                times.append(f.result())
                
        # To measure worst-case contention, we take the maximum latency across workers
        return max(times) if times else float('inf')

def run_workload_profiles(candidate: Dict[str, Any], n: int, k: int, is_moe: bool, 
                          available_gpus: int, backend: str, dtype_family: str) -> Dict[int, Dict[str, float]]:
    """
    Benchmarks the candidate across decode and prefill profiles (batch buckets).
    Returns a mapping of M (batch size) to latency results (both local and parallel).
    """
    decode_buckets = [1, 16]
    prefill_buckets = [64, 256]
    
    all_buckets = decode_buckets + prefill_buckets
    results = {}
    
    for m in all_buckets:
        local_time = BenchmarkModes.local_mode(candidate, m, n, k, backend, dtype_family, is_moe)
        
        # If the candidate crashes or performs abysmally in isolation, 
        # it will not survive parallel contention.
        if local_time == float('inf'):
            parallel_time = float('inf')
        else:
            parallel_time = BenchmarkModes.parallel_contention_mode(candidate, m, n, k, available_gpus, backend, dtype_family, is_moe)
            
        results[m] = {
            "local_ms": local_time,
            "parallel_ms": parallel_time
        }
        
    return results
