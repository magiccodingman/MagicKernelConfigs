import sys
import os
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any

from tuning.kernel_harness import get_harness_code


def _build_backend_env(backend: str, dtype_family: str, is_moe: bool) -> dict:
    env = os.environ.copy()

    project_root = str(Path(__file__).resolve().parents[1])

    existing = env.get("PYTHONPATH", "")
    paths = [project_root]

    if existing:
        paths.append(existing)

    env["PYTHONPATH"] = ":".join(paths)

    if backend == "aiter_triton":
        env["MAGIC_AITER_RUNNER"] = "tuning.aiter_runner:run_aiter_triton_kernel"

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
        if callable_name:
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

    return env


def validate_correctness(
    candidate: Dict[str, Any],
    n: int,
    k: int,
    m: int,
    dtype_family: str,
    is_moe: bool,
    backend: str
) -> bool:
    """
    Validates a kernel candidate by running it in an isolated subprocess.

    Enforces:
    - kernel executes successfully
    - no NaN / Inf
    - numeric closeness vs reference
    - backend used matches requested backend
    - backend proof is returned
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

            if "{dtype_family}" == "fp8":
                baseline_out = torch.matmul(a.to(torch.float16), b.to(torch.float16))
            else:
                baseline_out = torch.matmul(a_16, b_16)

            kernel_out = run_backend_kernel(a, b, candidate_params, "{backend}")

            if torch.isnan(kernel_out).any():
                raise RuntimeError("NaN detected in kernel output")

            if torch.isinf(kernel_out).any():
                raise RuntimeError("Inf detected in kernel output")

            if not torch.allclose(baseline_out, kernel_out, atol=1e-2):
                raise RuntimeError("Numeric discrepancy vs baseline")

            print(json.dumps({{
                "ok": True,
                "requested_backend": "{backend}",
                "actual_backend": ACTUAL_BACKEND,
                "backend_proof": BACKEND_PROOF
            }}))
            sys.exit(0)

        except Exception as e:
            print(json.dumps({{
                "ok": False,
                "requested_backend": "{backend}",
                "actual_backend": ACTUAL_BACKEND,
                "backend_proof": BACKEND_PROOF,
                "error": str(e)
            }}), file=sys.stderr)
            sys.exit(1)
    """)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness_code)
        temp_path = Path(f.name)

    try:
        env = _build_backend_env(backend, dtype_family, is_moe)

        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=20,
            env=env
        )

        if result.returncode != 0:
            print("\n[VALIDATOR FAILURE]")
            print("STDERR:")
            print(result.stderr.strip() or "<empty>")
            print("\nSTDOUT:")
            print(result.stdout.strip() or "<empty>")
            return False

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        if not lines:
            print("\n[VALIDATOR FAILURE] No stdout returned")
            print("STDERR:", result.stderr)
            return False

        try:
            payload = json.loads(lines[-1])
        except Exception:
            print("\n[VALIDATOR FAILURE] JSON parse error")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        if not payload.get("ok", False):
            print("\n[VALIDATOR PAYLOAD FAILURE]")
            print(payload)
            return False

        if payload.get("requested_backend") != backend:
            print("\n[BACKEND MISMATCH: requested]")
            print(payload)
            return False

        if payload.get("actual_backend") != backend:
            print("\n[BACKEND MISMATCH: actual]")
            print(payload)
            return False

        proof = payload.get("backend_proof", "")
        if proof:
            print(f"[Backend Proof] {proof}")

        # Extra safety for AITER callable proof
        if backend == "aiter_triton":
            expected_callable = env.get("MAGIC_AITER_CALLABLE", "")
            if expected_callable and expected_callable not in proof:
                print("\n[BACKEND PROOF MISMATCH]")
                print(f"Expected callable substring: {expected_callable}")
                print(f"Actual proof: {proof}")
                return False

        return True

    except subprocess.TimeoutExpired:
        print("\n[VALIDATOR TIMEOUT]")
        return False

    finally:
        temp_path.unlink(missing_ok=True)


def validate_minimal_runtime(json_config_path: Path, tp_max: int) -> bool:
    """
    Verifies JSON config schema compatibility with vLLM kernel loader expectations.
    """
    harness = textwrap.dedent(f"""\
        import sys
        import json

        try:
            with open(r'{json_config_path}', 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise RuntimeError("Config root must be a dict")

            for batch_str, params in data.items():
                batch = int(batch_str)

                if not isinstance(params, dict):
                    raise RuntimeError("Params must be dict")

                required = ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps"]

                for r in required:
                    if r not in params:
                        raise RuntimeError(f"Missing {{r}} for batch {{batch}}")

            sys.exit(0)

        except Exception as e:
            print(f"Runtime JSON validation failure: {{e}}", file=sys.stderr)
            sys.exit(1)
    """)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        temp_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print("\n[JSON VALIDATION FAILURE]")
            print(result.stderr.strip())
            return False

        return True

    except subprocess.TimeoutExpired:
        print("\n[JSON VALIDATION TIMEOUT]")
        return False

    finally:
        temp_path.unlink(missing_ok=True)