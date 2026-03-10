import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any

def run_isolated_benchmark_on_gpu(candidate: Dict[str, Any], m: int, n: int, k: int, gpu_id: str) -> float:
    """
    Spawns a process to benchmark the kernel on a specific GPU.
    Returns the execution time in milliseconds.
    """
    harness_code = textwrap.dedent(f"""\
        import sys
        import time
        import torch
        import os
        
        # Isolate to specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
        os.environ["ROCR_VISIBLE_DEVICES"] = "{gpu_id}"
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
            b = torch.randn(({k}, {n}), device=device, dtype=torch.float16)
            
            # Simulated kernel call: warmup
            for _ in range(3):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Simulated kernel call: timing
            start = time.perf_counter()
            iters = 10
            for _ in range(iters):
                torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            avg_ms = ((end - start) / iters) * 1000.0
            print(f"{{avg_ms:.5f}}")
            sys.exit(0)
            
        except Exception as e:
            print("-1.0", file=sys.stderr)
            sys.exit(1)
    """)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(harness_code)
        temp_path = Path(f.name)
        
    try:
        # A real benchmark wrapper would execute the specified vLLM tuning scripts here
        result = subprocess.run([sys.executable, str(temp_path)], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            try:
                # The output is expected to be the float latency
                lines = result.stdout.strip().split()
                if not lines:
                    return float('inf')
                return float(lines[-1])
            except ValueError:
                return float('inf')
        return float('inf')
    except subprocess.TimeoutExpired:
        return float('inf')
    finally:
        temp_path.unlink(missing_ok=True)

class BenchmarkModes:
    @staticmethod
    def local_mode(candidate: Dict[str, Any], m: int, n: int, k: int) -> float:
        """Evaluates candidate kernel in isolation on a single GPU."""
        return run_isolated_benchmark_on_gpu(candidate, m, n, k, "0")

    @staticmethod
    def parallel_contention_mode(candidate: Dict[str, Any], m: int, n: int, k: int, gpus: int) -> float:
        """
        Executes candidate kernels concurrently across available GPUs.
        Simulates tensor-parallel contention.
        """
        import concurrent.futures
        
        if gpus <= 1:
            return BenchmarkModes.local_mode(candidate, m, n, k)
            
        times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpus) as executor:
            futures = []
            for i in range(gpus):
                # We assume available GPUs are uniquely indexed starting from 0,
                # as the orchestrator isolated the system to only identical GPUs.
                futures.append(executor.submit(run_isolated_benchmark_on_gpu, candidate, m, n, k, str(i)))
                
            for f in concurrent.futures.as_completed(futures):
                times.append(f.result())
                
        # To measure worst-case contention, we take the maximum latency across workers
        return max(times) if times else float('inf')

def run_workload_profiles(candidate: Dict[str, Any], n: int, k: int, is_moe: bool, 
                          available_gpus: int) -> Dict[int, Dict[str, float]]:
    """
    Benchmarks the candidate across decode and prefill profiles (batch buckets).
    Returns a mapping of M (batch size) to latency results (both local and parallel).
    """
    decode_buckets = [1, 16]
    prefill_buckets = [64, 256]
    
    all_buckets = decode_buckets + prefill_buckets
    results = {}
    
    for m in all_buckets:
        local_time = BenchmarkModes.local_mode(candidate, m, n, k)
        
        # If the candidate crashes or performs abysmally in isolation, 
        # it will not survive parallel contention.
        if local_time == float('inf'):
            parallel_time = float('inf')
        else:
            parallel_time = BenchmarkModes.parallel_contention_mode(candidate, m, n, k, available_gpus)
            
        results[m] = {
            "local_ms": local_time,
            "parallel_ms": parallel_time
        }
        
    return results
