import importlib
import inspect
import os


def probe_aiter_triton_symbols():
    import aiter
    import aiter.ops.triton as ops

    names = sorted(dir(ops))
    gemmish = [n for n in names if "gemm" in n.lower() or "mm" in n.lower()]
    return gemmish

def probe_aiter_triton_signatures():
    out = {}

    candidates = [
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale:gemm_a8w8_blockscale",
        "aiter.ops.triton.gemm.basic.gemm_a8w8:gemm_a8w8",
        "aiter.ops.triton.gemm.basic.gemm_afp4wfp4:gemm_afp4wfp4",
        "aiter.ops.triton.gemm.basic.gemm_a8wfp4:gemm_a8wfp4",
        "aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale:moe_gemm_a8w8_blockscale",
        "aiter.ops.triton.moe.moe_op_gemm_a8w8:moe_gemm_a8w8",
        "aiter.ops.triton.moe.moe_op_gemm_a8w4:moe_gemm_a8w4",
        "aiter.ops.triton.moe.moe_op_gemm_a4w4:moe_gemm_a4w4",
    ]

    for spec in candidates:
        try:
            fn, proof = _resolve_aiter_callable(spec)
            try:
                out[proof] = str(inspect.signature(fn))
            except Exception:
                out[proof] = "<signature unavailable>"
        except Exception as e:
            out[spec] = f"<resolve failed: {e}>"

    return out


def _resolve_aiter_callable(spec: str | None = None):
    """
    Resolve the real AITER callable from:
      package.module:function_name

    Example:
      aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale:gemm_a8w8_blockscale

    Returns:
      (callable_obj, "package.module:function_name")
    """
    if spec is None:
        spec = os.environ.get("MAGIC_AITER_CALLABLE", "").strip()

    if not spec:
        raise RuntimeError(
            "MAGIC_AITER_CALLABLE is not set. "
            "Expected format: package.module:function_name"
        )

    if ":" not in spec:
        raise RuntimeError(
            f"Invalid MAGIC_AITER_CALLABLE '{spec}'. "
            "Expected format: package.module:function_name"
        )

    module_name, func_name = spec.split(":", 1)
    module_name = module_name.strip()
    func_name = func_name.strip()

    if not module_name or not func_name:
        raise RuntimeError(
            f"Invalid MAGIC_AITER_CALLABLE '{spec}'. "
            "Expected non-empty module and function names."
        )

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to import module '{module_name}' from MAGIC_AITER_CALLABLE '{spec}': {e}"
        ) from e

    if not hasattr(mod, func_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise RuntimeError(
            f"Callable '{func_name}' not found in module '{module_name}'. "
            f"Available exports: {available}"
        )

    fn = getattr(mod, func_name)

    if not callable(fn):
        raise RuntimeError(
            f"Resolved object '{func_name}' in '{module_name}' exists but is not callable "
            f"(type={type(fn)})."
        )

    proof = f"{module_name}:{func_name}"
    return fn, proof


def get_aiter_callable_proof() -> str:
    """
    Return the exact resolved AITER callable path for backend proof validation.
    """
    _, proof = _resolve_aiter_callable()
    return proof


def run_aiter_triton_kernel(a, b, candidate, requested_backend=None):
    import os
    import torch
    from tuning.aiter_fp8_adapter import make_aiter_fp8_blockscale_inputs

    if requested_backend is not None and requested_backend != "aiter_triton":
        raise RuntimeError(f"Invalid backend request: {requested_backend}")

    required = {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "VLLM_ROCM_USE_AITER_TRITON_GEMM": "1",
    }
    for key, expected in required.items():
        actual = os.environ.get(key)
        if actual != expected:
            raise RuntimeError(
                f"{key} must be {expected!r} for aiter_triton, got {actual!r}"
            )

    fn, proof = _resolve_aiter_callable()

    # Dense FP8 blockscale path
    x_int, w_int, x_scale, w_scale, _x_deq, _w_deq, config = make_aiter_fp8_blockscale_inputs(
        a.to(torch.float16),
        b.to(torch.float16),
        candidate,
    )

    M, K = x_int.shape
    N, K2 = w_int.shape
    if K != K2:
        raise RuntimeError(f"AITER adapted shape mismatch: x={tuple(x_int.shape)} w={tuple(w_int.shape)}")

    out = torch.empty((M, N), device=x_int.device, dtype=torch.float16)

    try:
        y = fn(
            x_int,
            w_int,
            x_scale,
            w_scale,
            dtype=torch.float16,
            y=out,
            config=config,
        )
        return y
    except Exception as e:
        raise RuntimeError(
            f"Resolved real AITER Triton callable ({proof}) but failed to invoke it. "
            f"x={tuple(x_int.shape)} w={tuple(w_int.shape)} "
            f"x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)} "
            f"config={config}. Last error: {e}"
        )