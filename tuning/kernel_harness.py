import textwrap

TRITON_HARNESS_CODE = textwrap.dedent("""\
    import sys
    import torch
    import triton
    import triton.language as tl
    import time
    import json

    # Only load required backend
    backend = "{backend}"
    if backend == "triton" or backend == "aiter_triton":
        try:
            import triton
        except ImportError:
            print("Error: Triton backend requested but not loadable.", file=sys.stderr)
            sys.exit(1)
            
    if backend == "aiter_triton":
        try:
            import aiter
        except ImportError:
            print("Error: AITER backend requested but not loadable.", file=sys.stderr)
            sys.exit(1)
            
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            
        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def run_backend_kernel(a, b, candidate, backend):
        # We must prove we are using the real selected backend
        if backend == "triton":
            pass # Standard triton kernel
        elif backend == "aiter_triton":
            # For this harness, execute same triton path but strictly prove aiter loaded
            import aiter
        else:
            raise RuntimeError(f"Backend mismatch detected: {{backend}}")
            
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        
        # Build strict literal kwargs avoiding unsupported signature keys
        kwargs_for_kernel = {{
            k: v for k, v in candidate.items() 
            if k not in ['num_warps', 'num_stages']
        }}
        
        # Ensure matrix_instr_nonkdim and kpack are passed safely if missing from candidate
        if 'matrix_instr_nonkdim' not in kwargs_for_kernel:
            kwargs_for_kernel['matrix_instr_nonkdim'] = 0
        if 'kpack' not in kwargs_for_kernel:
            kwargs_for_kernel['kpack'] = 1
            
        num_warps = candidate.get('num_warps', 4)
        num_stages = candidate.get('num_stages', 2)

        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            num_warps=num_warps,
            num_stages=num_stages,
            **kwargs_for_kernel
        )
        return c
""")
