import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any

from tuning.kernel_harness import get_harness_code

def validate_correctness(candidate: Dict[str, Any], n: int, k: int, m: int, dtype_family: str, is_moe: bool, backend: str) -> bool:
    """
    Validates a kernel candidate by running it in an isolated subprocess.
    Ensures safe kernel execution without crashing, checks tolerance limits,
    and prevents PyTorch/Triton segfaults from taking down the tuner.
    """
    harness_code = get_harness_code(backend, dtype_family, is_moe) + textwrap.dedent(f"""\
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if "{dtype_family}" == "fp8":
                a_16 = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
                b_16 = torch.randn(({k}, {n}), device=device, dtype=torch.float16)
                a = a_16.to(torch.float8_e4m3fn)
                b = b_16.to(torch.float8_e4m3fn)
            elif "{dtype_family}" == "int8":
                a_16 = torch.randint(-128, 127, ({m}, {k}), device=device, dtype=torch.int8).to(torch.float16)
                b_16 = torch.randint(-128, 127, ({k}, {n}), device=device, dtype=torch.int8).to(torch.float16)
                a = a_16.to(torch.int8)
                b = b_16.to(torch.int8)
            else:
                a_16 = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
                b_16 = torch.randn(({k}, {n}), device=device, dtype=torch.float16)
                a = a_16
                b = b_16
            
            candidate_params = {candidate}
            
            # Reference baseline via float16 exact computation
            if "{dtype_family}" == "fp8":
                baseline_out = torch.matmul(a.to(torch.float16), b.to(torch.float16))
            elif "{dtype_family}" == "int8":
                baseline_out = torch.matmul(a_16, b_16)
            else:
                baseline_out = torch.matmul(a_16, b_16)
            
            # Real Native Subprocess Compilation
            kernel_out = run_backend_kernel(a, b, candidate_params, "{backend}")
            
            if torch.isnan(kernel_out).any() or torch.isinf(kernel_out).any():
                print("NaN/Inf detected in output.", file=sys.stderr)
                sys.exit(1)
                
            # Tolerance checking
            if not torch.allclose(baseline_out, kernel_out, atol=1e-2):
                print("Numeric discrepancy vs reference.", file=sys.stderr)
                sys.exit(1)
                
            sys.exit(0)
            
        except Exception as e:
            print(f"Crash during correctness validation: {{e}}", file=sys.stderr)
            sys.exit(1)
    """)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(harness_code)
        temp_path = Path(f.name)
        
    try:
        result = subprocess.run([sys.executable, str(temp_path)], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        temp_path.unlink(missing_ok=True)
         
def validate_minimal_runtime(json_config_path: Path, tp_max: int) -> bool:
    """
    Validates that a generated JSON config file can be successfully loaded 
    by simulating vLLM's kernel configuration loader requirements.
    """
    harness = textwrap.dedent(f"""\
        import sys
        import json
        try:
            with open('{json_config_path}', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Must be keyed by batch size (integer strings)
            for batch_str, params in data.items():
                batch = int(batch_str)
                assert "BLOCK_SIZE_M" in params, "Missing BLOCK_SIZE_M"
                assert "num_warps" in params, "Missing num_warps"
                
            sys.exit(0)
        except Exception as e:
            print(f"Runtime JSON parsing/schema failure: {{e}}", file=sys.stderr)
            sys.exit(1)
    """)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(harness)
        temp_path = Path(f.name)
        
    try:
        result = subprocess.run([sys.executable, str(temp_path)], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        temp_path.unlink(missing_ok=True)
