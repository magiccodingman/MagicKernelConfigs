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

    # Ensure the temp subprocess can import this project
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
        if callable_name:
            env["MAGIC_AITER_CALLABLE"] = callable_name

        env["VLLM_ROCM_USE_AITER"] = "1"
        env["VLLM_ROCM_USE_AITER_LINEAR"] = "1"
        env["VLLM_ROCM_USE_AITER_TRITON_GEMM"] = "1"

        # Disable unrelated AITER subsystems so we isolate GEMM tuning
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
        from tuning.aiter_fp8_adapter import make_aiter_fp8_blockscale_inputs

        try:
            torch.manual_seed(777)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(777)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            candidate_params = {candidate}

            a_16 = torch.randn(({m}, {k}), device=device, dtype=torch.float16)
            b_16 = torch.randn(({k}, {n}), device=device, dtype=torch.float16)

            if "{backend}" == "triton":
                if "{dtype_family}" == "fp8":
                    a = a_16.to(torch.float8_e4m3fn)
                    b = b_16.to(torch.float8_e4m3fn)
                    baseline_out = torch.matmul(a.to(torch.float16), b.to(torch.float16))

                elif "{dtype_family}" == "int8":
                    a = a_16.round().clamp(-128, 127).to(torch.int8)
                    b = b_16.round().clamp(-128, 127).to(torch.int8)
                    baseline_out = torch.matmul(a_16, b_16)

                else:
                    a = a_16
                    b = b_16
                    baseline_out = torch.matmul(a_16, b_16)

            elif "{backend}" == "aiter_triton":
                # Keep float16 masters; AITER adapter quantizes/adapts internally.
                a = a_16
                b = b_16

                if "{dtype_family}" != "fp8":
                    raise RuntimeError("aiter_triton correctness is only implemented for fp8 right now")

                x_int, w_int, x_scale, w_scale, x_deq, w_deq, config = make_aiter_fp8_blockscale_inputs(
                    a_16,
                    b_16,
                    candidate_params
                )

                # AITER computes Y = X @ W^T, where W is stored as (N, K)
                baseline_out = torch.matmul(x_deq, w_deq.transpose(0, 1)).to(torch.float16)

            else:
                raise RuntimeError("Unsupported backend in validator: {backend}")

            kernel_out = run_backend_kernel(a, b, candidate_params, "{backend}")

            if torch.isnan(kernel_out).any():
                raise RuntimeError("NaN detected in kernel output")

            if torch.isinf(kernel_out).any():
                raise RuntimeError("Inf detected in kernel output")

            ref = baseline_out.to(torch.float32)
            out = kernel_out.to(torch.float32)

            abs_diff = (ref - out).abs()
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()

            denom = ref.abs().clamp_min(1e-5)
            max_rel_diff = (abs_diff / denom).max().item()

            if "{backend}" == "aiter_triton":
                # Quantized blockscale path needs looser tolerance than raw Triton path
                atol = 5e-1
                rtol = 5e-2
            else:
                atol = 1e-2
                rtol = 1e-2

            if not torch.allclose(ref, out, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"Numeric discrepancy vs baseline | "
                    f"max_abs_diff={{max_abs_diff:.6f}} "
                    f"mean_abs_diff={{mean_abs_diff:.6f}} "
                    f"max_rel_diff={{max_rel_diff:.6f}} "
                    f"atol={{atol}} rtol={{rtol}}"
                )

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
            print("STDERR:", result.stderr.strip() or "<empty>")
            print("STDOUT:", result.stdout.strip() or "<empty>")
            return False

        try:
            payload = json.loads(lines[-1])
        except Exception:
            print("\n[VALIDATOR FAILURE] JSON parse error")
            print("STDOUT:", result.stdout.strip() or "<empty>")
            print("STDERR:", result.stderr.strip() or "<empty>")
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
    Verifies JSON config schema compatibility with the generated kernel config expectations.

    We allow either:
    - Triton-style batch entries with num_warps
    - AITER-style batch entries with NUM_KSPLIT
    """
    harness = textwrap.dedent(f"""\
            import sys
            import json

            try:
                with open(r'{json_config_path}', 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    raise RuntimeError("Config root must be a dict")

                allowed = {{
                    "BLOCK_SIZE_M",
                    "BLOCK_SIZE_N",
                    "BLOCK_SIZE_K",
                    "GROUP_SIZE_M",
                    "num_warps",
                    "num_stages",
                    "kpack",
                    "matrix_instr_nonkdim",
                }}

                required_common = ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]

                for batch_str, params in data.items():
                    batch = int(batch_str)

                    if not isinstance(params, dict):
                        raise RuntimeError(f"Params for batch {{batch}} must be dict")

                    for r in required_common:
                        if r not in params:
                            raise RuntimeError(f"Missing {{r}} for batch {{batch}}")

                    if "NUM_KSPLIT" in params:
                        raise RuntimeError(
                            f"Batch {{batch}} contains NUM_KSPLIT, which is not allowed in persisted vLLM configs"
                        )

                    if "num_warps" not in params:
                        raise RuntimeError(f"Missing num_warps for batch {{batch}}")

                    if "num_stages" not in params:
                        raise RuntimeError(f"Missing num_stages for batch {{batch}}")

                    unknown = [k for k in params.keys() if k not in allowed]
                    if unknown:
                        raise RuntimeError(
                            f"Unknown keys for batch {{batch}}: {{unknown}}"
                        )

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
            print(result.stderr.strip() or "<empty>")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("\n[JSON VALIDATION TIMEOUT]")
        return False

    finally:
        temp_path.unlink(missing_ok=True)