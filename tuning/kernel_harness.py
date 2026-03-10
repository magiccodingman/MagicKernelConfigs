import textwrap


def _base_prelude(backend: str, dtype_family: str, is_moe: bool) -> str:
    return textwrap.dedent(f"""\
        import os
        import sys
        import json
        import time
        import importlib
        import torch

        backend = "{backend}"
        dtype_family = "{dtype_family}"
        is_moe = {is_moe}

        ACTUAL_BACKEND = None
        BACKEND_PROOF = None

        def _emit_success(payload: dict):
            print(json.dumps(payload))
            sys.exit(0)

        def _emit_error(msg: str):
            print(json.dumps({{
                "ok": False,
                "requested_backend": backend,
                "actual_backend": ACTUAL_BACKEND,
                "backend_proof": BACKEND_PROOF,
                "error": msg
            }}), file=sys.stderr)
            sys.exit(1)

        def _import_callable(spec: str):
            '''
            spec format:
              package.module:function_name

            Example:
              mypkg.runners:run_aiter_triton_kernel
            '''
            if ":" not in spec:
                raise RuntimeError(
                    "MAGIC_AITER_RUNNER must look like 'package.module:function_name'"
                )

            mod_name, func_name = spec.split(":", 1)
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name)
            if not callable(fn):
                raise RuntimeError(f"Resolved object is not callable: {{spec}}")
            return fn
    """)


def _build_triton_harness(dtype_family: str, is_moe: bool) -> str:
    base = _base_prelude("triton", dtype_family, is_moe)

    if dtype_family == "fp8":
        kernel_def = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def benchmark_kernel(
                a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr,
                matrix_instr_nonkdim: tl.constexpr = 0,
                kpack: tl.constexpr = 1
            ):
                pid = tl.program_id(axis=0)

                num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
                num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
                num_pid_in_group = GROUP_SIZE_M * num_pid_n

                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk
                    K -= BLOCK_SIZE_K

                c = accumulator.to(tl.float16)
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)
        """)

    elif dtype_family == "int8":
        kernel_def = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def benchmark_kernel(
                a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr,
                matrix_instr_nonkdim: tl.constexpr = 0,
                kpack: tl.constexpr = 1
            ):
                pid = tl.program_id(axis=0)

                num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
                num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
                num_pid_in_group = GROUP_SIZE_M * num_pid_n

                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

                for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk
                    K -= BLOCK_SIZE_K

                c = accumulator.to(tl.float16)
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)
        """)

    else:
        raise RuntimeError(f"Unsupported dtype_family for Triton harness: {dtype_family}")

    runner_def = textwrap.dedent("""\
        def run_backend_kernel(a, b, candidate, requested_backend):
            global ACTUAL_BACKEND, BACKEND_PROOF

            if requested_backend != "triton":
                raise RuntimeError(
                    f"Triton harness received wrong requested backend: {requested_backend}"
                )

            if not torch.cuda.is_available():
                raise RuntimeError("Triton kernel requires GPU device")

            M, K = a.shape
            K2, N = b.shape
            if K != K2:
                raise RuntimeError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

            c = torch.empty((M, N), device=a.device, dtype=torch.float16)

            kwargs = {k: v for k, v in candidate.items() if k not in ["num_warps", "num_stages"]}

            if "matrix_instr_nonkdim" not in kwargs:
                kwargs["matrix_instr_nonkdim"] = 0
            if "kpack" not in kwargs:
                kwargs["kpack"] = 1

            num_warps = candidate.get("num_warps", 4)
            num_stages = candidate.get("num_stages", 2)

            grid = lambda META: (
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

            benchmark_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                num_warps=num_warps,
                num_stages=num_stages,
                **kwargs
            )

            ACTUAL_BACKEND = "triton"
            BACKEND_PROOF = "custom_triton_jit_kernel"

            return c
    """)

    return base + "\n" + kernel_def + "\n" + runner_def


def _build_aiter_triton_harness(dtype_family: str, is_moe: bool) -> str:
    base = _base_prelude("aiter_triton", dtype_family, is_moe)

    runner_def = textwrap.dedent("""\
        try:
            import aiter
            import aiter.ops.triton
        except Exception as e:
            raise RuntimeError(
                f"AITER Triton backend requested but aiter/aiter.ops.triton failed to import: {e}"
            )

        def _resolve_aiter_runner():
            '''
            You MUST point this at a real AITER execution callable.

            Set environment variable:
              MAGIC_AITER_RUNNER=package.module:function_name

            Example:
              export MAGIC_AITER_RUNNER=myproject.aiter_runner:run_aiter_triton_kernel

            The callable must accept either:
              fn(a, b, candidate)
            or:
              fn(a, b, candidate, requested_backend)
            '''
            spec = os.environ.get("MAGIC_AITER_RUNNER", "").strip()
            if not spec:
                raise RuntimeError(
                    "aiter_triton requested, but no real AITER callable is configured. "
                    "Set MAGIC_AITER_RUNNER=package.module:function_name"
                )
            return _import_callable(spec), spec

        def run_backend_kernel(a, b, candidate, requested_backend):
            global ACTUAL_BACKEND, BACKEND_PROOF

            if requested_backend != "aiter_triton":
                raise RuntimeError(
                    f"AITER-Triton harness received wrong requested backend: {requested_backend}"
                )

            runner, proof = _resolve_aiter_runner()

            try:
                out = runner(a, b, candidate)
            except TypeError:
                out = runner(a, b, candidate, requested_backend)

            ACTUAL_BACKEND = "aiter_triton"
            BACKEND_PROOF = f"external_aiter_runner:{proof}"
            return out
    """)

    return base + "\n" + runner_def


def get_harness_code(backend: str, dtype_family: str, is_moe: bool) -> str:
    """
    Returns the Python source code for the isolated subprocess harness.

    Rules:
    - backend="triton"        => real Triton harness
    - backend="aiter_triton"  => ONLY a real AITER-Triton callable is allowed
    - pure AITER is not supported here
    """
    if backend == "triton":
        return _build_triton_harness(dtype_family, is_moe)

    if backend == "aiter_triton":
        return _build_aiter_triton_harness(dtype_family, is_moe)

    raise RuntimeError(f"Unsupported backend for harness generation: {backend}")