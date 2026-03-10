import itertools
from typing import List, Dict, Any


def _dedup_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup = {frozenset(d.items()): d for d in items}
    return list(dedup.values())


def _shape_bucket_n(target_n: int) -> str:
    if target_n >= 2048:
        return "large"
    if target_n >= 512:
        return "medium"
    return "small"


def _shape_bucket_k(target_k: int) -> str:
    if target_k >= 4096:
        return "huge"
    if target_k >= 1024:
        return "large"
    if target_k >= 256:
        return "medium"
    return "small"


def _vendor_group_sizes(vendor: str) -> List[int]:
    """
    Conservative, vendor-aware defaults.

    GROUP_SIZE_M beyond 8 often gives diminishing returns or pointless search
    explosion for public AITER/triton GEMM surfaces.
    """
    vendor = (vendor or "").lower()

    if vendor == "amd":
        return [1, 4, 8]
    if vendor == "nvidia":
        return [1, 4, 8]
    if vendor == "intel":
        return [1, 2, 4, 8]

    # unknown vendor: stay conservative
    return [1, 4, 8]


def _triton_warps_for_vendor(vendor: str) -> List[int]:
    vendor = (vendor or "").lower()

    if vendor == "amd":
        return [2, 4, 8]
    if vendor == "nvidia":
        return [2, 4, 8]
    if vendor == "intel":
        return [2, 4]

    return [2, 4, 8]


def _triton_dense_shape_space(vendor: str, target_n: int, target_k: int) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Shape-aware Triton space. Still flexible, but avoids obviously dumb configs.
    """
    n_bucket = _shape_bucket_n(target_n)
    k_bucket = _shape_bucket_k(target_k)

    # BLOCK_SIZE_M
    # 16 is mainly useful for tiny/decode-ish shapes; otherwise it is often too small.
    if n_bucket == "small":
        m_blocks = [16, 32, 64]
    else:
        m_blocks = [32, 64, 128]

    # BLOCK_SIZE_N
    if n_bucket == "large":
        n_blocks = [64, 128, 256]
    elif n_bucket == "medium":
        n_blocks = [32, 64, 128]
    else:
        n_blocks = [16, 32, 64]

    # BLOCK_SIZE_K
    if k_bucket == "huge":
        k_blocks = [64, 128]
    elif k_bucket == "large":
        k_blocks = [32, 64, 128]
    elif k_bucket == "medium":
        k_blocks = [32, 64]
    else:
        k_blocks = [16, 32, 64]

    g_blocks = _vendor_group_sizes(vendor)
    warps = _triton_warps_for_vendor(vendor)

    return m_blocks, n_blocks, k_blocks, g_blocks, warps


def _aiter_triton_fp8_shape_space(vendor: str, target_n: int, target_k: int) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Public AITER dense FP8 blockscale config surface:
      BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K / GROUP_SIZE_M / NUM_KSPLIT

    This is intentionally much tighter than Triton, because AITER public wrappers
    do not expose Triton-internal knobs like num_warps / kpack / matrix_instr_nonkdim.
    """
    n_bucket = _shape_bucket_n(target_n)
    k_bucket = _shape_bucket_k(target_k)

    # BLOCK_SIZE_M
    # Keep 16 only for smaller-N cases. For large shapes it is usually wasted search.
    if n_bucket == "small":
        m_blocks = [16, 32, 64]
    else:
        m_blocks = [32, 64, 128]

    # BLOCK_SIZE_N
    if n_bucket == "large":
        n_blocks = [64, 128, 256]
    elif n_bucket == "medium":
        n_blocks = [32, 64, 128]
    else:
        n_blocks = [16, 32, 64]

    # BLOCK_SIZE_K
    # For large K, 64/128 dominate. Keep 32 only when K is not giant.
    if k_bucket == "huge":
        k_blocks = [64, 128]
    elif k_bucket == "large":
        k_blocks = [32, 64, 128]
    elif k_bucket == "medium":
        k_blocks = [32, 64]
    else:
        k_blocks = [16, 32, 64]

    # GROUP_SIZE_M
    g_blocks = _vendor_group_sizes(vendor)

    # NUM_KSPLIT
    # Keep this conservative by default. Large-K cases may benefit from 2.
    if k_bucket == "huge":
        ksplits = [1, 2]
    else:
        ksplits = [1]

    return m_blocks, n_blocks, k_blocks, g_blocks, ksplits


def get_base_candidates_triton_dense(vendor: str, target_n: int, target_k: int) -> List[Dict[str, Any]]:
    m_blocks, n_blocks, k_blocks, g_blocks, warps = _triton_dense_shape_space(
        vendor=vendor,
        target_n=target_n,
        target_k=target_k,
    )

    out = []
    for m, n, k, g, w in itertools.product(m_blocks, n_blocks, k_blocks, g_blocks, warps):
        out.append({
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": n,
            "BLOCK_SIZE_K": k,
            "GROUP_SIZE_M": g,
            "num_warps": w,
        })
    return out


def get_base_candidates_aiter_triton_dense_fp8(vendor: str, target_n: int, target_k: int) -> List[Dict[str, Any]]:
    m_blocks, n_blocks, k_blocks, g_blocks, ksplits = _aiter_triton_fp8_shape_space(
        vendor=vendor,
        target_n=target_n,
        target_k=target_k,
    )

    out = []
    for m, n, k, g, ks in itertools.product(m_blocks, n_blocks, k_blocks, g_blocks, ksplits):
        out.append({
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": n,
            "BLOCK_SIZE_K": k,
            "GROUP_SIZE_M": g,
            "NUM_KSPLIT": ks,
        })
    return out


def filter_occupancy(
    candidate: Dict[str, Any],
    is_moe: bool,
    backend: str,
    vendor: str,
    target_n: int,
    target_k: int,
) -> bool:
    """
    Cross-backend, cross-vendor pruning.

    This is not meant to perfectly model hardware; it's meant to remove obviously
    unproductive search points without throwing away realistic winners.
    """
    vendor = (vendor or "").lower()
    backend = (backend or "").lower()

    m = int(candidate["BLOCK_SIZE_M"])
    n = int(candidate["BLOCK_SIZE_N"])
    k = int(candidate["BLOCK_SIZE_K"])
    w = int(candidate.get("num_warps", 4))
    g = int(candidate.get("GROUP_SIZE_M", 1))
    ks = int(candidate.get("NUM_KSPLIT", 1))

    block_vol = m * n * k

    # Hard upper bound: absurdly large tiles
    if block_vol > 128 * 256 * 128:
        return False

    # Tiny tiles are almost never worth it on modern GPUs for medium/large shapes
    if target_n >= 512 and m < 32:
        return False
    if target_n >= 512 and n < 32:
        return False

    # For very large N, 16-wide N tiles are essentially wasted exploration
    if target_n >= 2048 and n < 64:
        return False

    # For very large K, tiny K tiles become mostly overhead
    if target_k >= 4096 and k < 64:
        return False

    # group size beyond 8 is usually not worth the extra search for dense public APIs
    if g > 8:
        return False

    # NUM_KSPLIT > 1 is only reasonable when K is big enough to justify it
    if ks > 1 and target_k < 2048:
        return False

    # Vendor-aware warp sanity
    if backend == "triton":
        if vendor == "intel" and w > 4:
            return False

        # very small output tiles with huge warps are wasteful
        if m * n < 1024 and w > 4:
            return False

    # AITER public FP8 blockscale path does not use Triton-only internal knobs
    if backend == "aiter_triton":
        if "num_warps" in candidate:
            return False
        if "kpack" in candidate:
            return False
        if "matrix_instr_nonkdim" in candidate:
            return False

    if is_moe:
        stages = int(candidate.get("num_stages", 2))
        if stages >= 4 and block_vol > 64 * 128 * 64:
            return False

    return True


class DenseCandidateBuilder:
    def __init__(
        self,
        is_amd: bool,
        is_fp8: bool,
        backend: str,
        vendor: str,
        target_n: int,
        target_k: int,
    ):
        self.is_amd = is_amd
        self.is_fp8 = is_fp8
        self.backend = backend
        self.vendor = vendor
        self.target_n = target_n
        self.target_k = target_k

    def build(self) -> List[Dict[str, Any]]:
        backend = (self.backend or "").lower()

        # Separate backend-specific candidate spaces
        if backend == "aiter_triton":
            if not self.is_fp8:
                raise RuntimeError("aiter_triton dense builder is currently implemented for fp8 only")

            base = get_base_candidates_aiter_triton_dense_fp8(
                vendor=self.vendor,
                target_n=self.target_n,
                target_k=self.target_k,
            )

            results = [
                c for c in base
                if filter_occupancy(
                    c,
                    is_moe=False,
                    backend=backend,
                    vendor=self.vendor,
                    target_n=self.target_n,
                    target_k=self.target_k,
                )
            ]
            return _dedup_dicts(results)

        # Triton path
        base = get_base_candidates_triton_dense(
            vendor=self.vendor,
            target_n=self.target_n,
            target_k=self.target_k,
        )

        results = []

        for p in base:
            # AMD FP8 Triton path can expose extra internal search knobs
            if self.is_amd and self.is_fp8 and backend == "triton":
                for kpack in [1, 2]:
                    for nonkdim in [16, 32]:
                        c = p.copy()
                        c["kpack"] = kpack
                        c["matrix_instr_nonkdim"] = nonkdim
                        if filter_occupancy(
                            c,
                            is_moe=False,
                            backend=backend,
                            vendor=self.vendor,
                            target_n=self.target_n,
                            target_k=self.target_k,
                        ):
                            results.append(c)
            else:
                if filter_occupancy(
                    p,
                    is_moe=False,
                    backend=backend,
                    vendor=self.vendor,
                    target_n=self.target_n,
                    target_k=self.target_k,
                ):
                    results.append(p)

        return _dedup_dicts(results)


class MoECandidateBuilder:
    def __init__(
        self,
        backend: str,
        vendor: str,
        target_n: int,
        target_k: int,
    ):
        self.backend = backend
        self.vendor = vendor
        self.target_n = target_n
        self.target_k = target_k

    def build(self) -> List[Dict[str, Any]]:
        backend = (self.backend or "").lower()

        if backend == "aiter_triton":
            # Honest fail-closed for now. AITER public MoE wrapper computes its own config surface.
            raise RuntimeError("aiter_triton MoE tuning is not implemented yet")

        base = get_base_candidates_triton_dense(
            vendor=self.vendor,
            target_n=self.target_n,
            target_k=self.target_k,
        )

        results = []
        stages_options = [2, 3, 4, 5]

        for p in base:
            for stages in stages_options:
                c = p.copy()
                c["num_stages"] = stages
                if filter_occupancy(
                    c,
                    is_moe=True,
                    backend=backend,
                    vendor=self.vendor,
                    target_n=self.target_n,
                    target_k=self.target_k,
                ):
                    results.append(c)

        return _dedup_dicts(results)


def mutate_candidate(
    candidate: Dict[str, Any],
    is_moe: bool,
    backend: str,
    vendor: str,
    target_n: int,
    target_k: int,
) -> List[Dict[str, Any]]:
    mutations = []
    backend = (backend or "").lower()

    if backend == "triton":
        # Triton-specific mutation surface
        for w in _triton_warps_for_vendor(vendor):
            if w != candidate.get("num_warps", -1):
                c = candidate.copy()
                c["num_warps"] = w
                mutations.append(c)

        for dim in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]:
            if dim in candidate:
                current = candidate[dim]
                for adj in [current // 2, current * 2]:
                    if 16 <= adj <= 256 and adj != current:
                        c = candidate.copy()
                        c[dim] = adj
                        mutations.append(c)

        if "GROUP_SIZE_M" in candidate:
            gs = candidate["GROUP_SIZE_M"]
            for adj in [1, 4, 8]:
                if adj != gs:
                    c = candidate.copy()
                    c["GROUP_SIZE_M"] = adj
                    mutations.append(c)

        if "kpack" in candidate:
            kp = candidate["kpack"]
            for adj in [1, 2]:
                if adj != kp:
                    c = candidate.copy()
                    c["kpack"] = adj
                    mutations.append(c)

        if "matrix_instr_nonkdim" in candidate:
            nd = candidate["matrix_instr_nonkdim"]
            for adj in [16, 32]:
                if adj != nd:
                    c = candidate.copy()
                    c["matrix_instr_nonkdim"] = adj
                    mutations.append(c)

    elif backend == "aiter_triton":
        # Only mutate along the public AITER config surface
        for dim in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]:
            if dim in candidate:
                current = candidate[dim]
                for adj in [current // 2, current * 2]:
                    if 16 <= adj <= 256 and adj != current:
                        c = candidate.copy()
                        c[dim] = adj
                        mutations.append(c)

        if "GROUP_SIZE_M" in candidate:
            gs = candidate["GROUP_SIZE_M"]
            for adj in [1, 4, 8]:
                if adj != gs:
                    c = candidate.copy()
                    c["GROUP_SIZE_M"] = adj
                    mutations.append(c)

        if "NUM_KSPLIT" in candidate:
            ks = candidate["NUM_KSPLIT"]
            for adj in [1, 2]:
                if adj != ks:
                    c = candidate.copy()
                    c["NUM_KSPLIT"] = adj
                    mutations.append(c)

    valid_mutations = [
        m for m in mutations
        if filter_occupancy(
            m,
            is_moe=is_moe,
            backend=backend,
            vendor=vendor,
            target_n=target_n,
            target_k=target_k,
        )
    ]

    return _dedup_dicts(valid_mutations)