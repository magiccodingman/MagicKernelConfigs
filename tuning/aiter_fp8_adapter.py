import math
import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_aiter_fp8_blockscale_inputs(a_fp16: torch.Tensor,
                                     b_fp16: torch.Tensor,
                                     candidate: dict):
    """
    Build AITER FP8 blockscale inputs from float16 source tensors.

    Input expectations from the rest of the tuner:
      a_fp16: (M, K)
      b_fp16: (K, N)

    AITER dense FP8 blockscale expects:
      x: (M, K) int8
      w: (N, K) int8
      x_scale: (M, scale_k)
      w_scale: (scale_n, scale_k)
      config: BLOCK_SIZE_M/BLOCK_SIZE_N/BLOCK_SIZE_K/GROUP_SIZE_M/NUM_KSPLIT

    Returns:
      x_int, w_int, x_scale, w_scale, x_deq, w_deq, config
    """
    if a_fp16.dtype != torch.float16:
        a_fp16 = a_fp16.to(torch.float16)
    if b_fp16.dtype != torch.float16:
        b_fp16 = b_fp16.to(torch.float16)

    M, K = a_fp16.shape
    K2, N = b_fp16.shape
    if K != K2:
        raise RuntimeError(f"Shape mismatch: a_fp16={tuple(a_fp16.shape)} b_fp16={tuple(b_fp16.shape)}")

    block_m = int(candidate["BLOCK_SIZE_M"])
    block_n = int(candidate["BLOCK_SIZE_N"])
    block_k = int(candidate["BLOCK_SIZE_K"])
    group_m = int(candidate["GROUP_SIZE_M"])
    num_ksplit = int(candidate.get("NUM_KSPLIT", 1))

    # AITER expects weights in (N, K)
    w_fp16 = b_fp16.transpose(0, 1).contiguous()

    scale_k = _ceil_div(K, block_k)
    scale_n = _ceil_div(N, block_n)

    x_int = torch.empty((M, K), device=a_fp16.device, dtype=torch.int8)
    x_deq = torch.empty((M, K), device=a_fp16.device, dtype=torch.float32)
    x_scale = torch.empty((M, scale_k), device=a_fp16.device, dtype=torch.float32)

    # Per-row, per-K-block scaling for x
    for kb in range(scale_k):
        k0 = kb * block_k
        k1 = min((kb + 1) * block_k, K)

        tile = a_fp16[:, k0:k1]  # (M, kblk)
        s = tile.abs().amax(dim=1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)

        q = torch.round(tile / s).clamp(-128, 127).to(torch.int8)
        dq = q.to(torch.float32) * s

        x_int[:, k0:k1] = q
        x_deq[:, k0:k1] = dq
        x_scale[:, kb] = s.squeeze(1)

    w_int = torch.empty((N, K), device=w_fp16.device, dtype=torch.int8)
    w_deq = torch.empty((N, K), device=w_fp16.device, dtype=torch.float32)
    w_scale = torch.empty((scale_n, scale_k), device=w_fp16.device, dtype=torch.float32)

    # Per-(N-block, K-block) scaling for w
    for nb in range(scale_n):
        n0 = nb * block_n
        n1 = min((nb + 1) * block_n, N)

        for kb in range(scale_k):
            k0 = kb * block_k
            k1 = min((kb + 1) * block_k, K)

            tile = w_fp16[n0:n1, k0:k1]
            s = tile.abs().amax() / 127.0
            if s == 0:
                s = torch.tensor(1.0, device=w_fp16.device, dtype=torch.float32)
            else:
                s = s.to(torch.float32)

            q = torch.round(tile / s).clamp(-128, 127).to(torch.int8)
            dq = q.to(torch.float32) * s

            w_int[n0:n1, k0:k1] = q
            w_deq[n0:n1, k0:k1] = dq
            w_scale[nb, kb] = s

    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "NUM_KSPLIT": num_ksplit,
    }

    return x_int, w_int, x_scale, w_scale, x_deq, w_deq, config