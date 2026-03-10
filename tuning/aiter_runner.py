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
    import aiter
    import aiter.ops.triton as ops

    out = {}
    for name in sorted(dir(ops)):
        if "gemm" not in name.lower() and "mm" not in name.lower():
            continue
        obj = getattr(ops, name)
        if callable(obj):
            try:
                out[name] = str(inspect.signature(obj))
            except Exception:
                out[name] = "<signature unavailable>"
    return out


def _resolve_aiter_callable():
    symbol = os.environ.get("MAGIC_AITER_CALLABLE", "").strip()
    if not symbol:
        raise RuntimeError(
            "MAGIC_AITER_CALLABLE is not set. "
            "Set it to a symbol from aiter.ops.triton, e.g. gemm_a8w8_blockscale"
        )

    import aiter
    import aiter.ops.triton as ops

    if not hasattr(ops, symbol):
        available = [n for n in dir(ops) if "gemm" in n.lower() or "mm" in n.lower()]
        raise RuntimeError(
            f"AITER Triton symbol '{symbol}' not found. Available candidates: {available}"
        )

    fn = getattr(ops, symbol)
    if not callable(fn):
        raise RuntimeError(f"AITER Triton symbol '{symbol}' exists but is not callable")

    return fn, f"aiter.ops.triton.{symbol}"


def run_aiter_triton_kernel(a, b, candidate, requested_backend=None):
    """
    Real AITER Triton entrypoint wrapper.
    """
    # Accept None for safety, but if provided it must be correct
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

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise RuntimeError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

    # Candidate invocation attempts
    last_err = None

    # Most likely public style
    try:
        out = fn(a, b, **candidate)
        return out
    except Exception as e:
        last_err = e

    # Some wrappers may want explicit backend
    try:
        out = fn(a, b, candidate, requested_backend)
        return out
    except Exception as e:
        last_err = e

    # Some wrappers may accept candidate dict only
    try:
        out = fn(a, b, candidate)
        return out
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"Resolved real AITER Triton callable ({proof}) but failed to invoke it. "
        f"Last error: {last_err}"
    )