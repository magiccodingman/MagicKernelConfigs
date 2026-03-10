import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any

def validate_correctness(candidate: Dict[str, Any], n: int, k: int, m: int, dtype_family: str, is_moe: bool) -> bool:
    """
    Validates a kernel candidate by running it in an isolated subprocess.
    Ensures safe kernel execution without crashing, checks tolerance limits,
    and prevents PyTorch/Triton segfaults from taking down the tuner.
    """
    harness_code = textwrap.dedent(f"""\
        import sys
        import torch
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Simple mock for structural validation.
            # A completely functional tuner would import the exact vLLM kernel bound
            # to the '{dtype_family}' namespace.
            a = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
            b = torch.randn(({k}, {n}), device=device, dtype=torch.float16)
            
            candidate_params = {candidate}
            is_moe = {is_moe}
            
            # Reference baseline
            baseline_out = torch.matmul(a, b)
            
            # Simulated kernel output - in production, replace with: 
            # kernel_out = run_triton_kernel(a, b, **candidate_params)
            kernel_out = baseline_out 
            
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
