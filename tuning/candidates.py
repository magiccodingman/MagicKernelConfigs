import itertools
from typing import List, Dict, Any


def get_base_candidates_triton_dense() -> List[Dict[str, Any]]:
    m_blocks = [16, 32, 64, 128]
    n_blocks = [16, 32, 64, 128, 256]
    k_blocks = [32, 64, 128, 256]
    g_blocks = [1, 8, 16, 32]
    warps = [2, 4, 8]

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


def get_base_candidates_aiter_triton_dense_fp8() -> List[Dict[str, Any]]:
    # Public AITER dense FP8 blockscale config surface:
    # BLOCK_SIZE_M/N/K, GROUP_SIZE_M, NUM_KSPLIT
    m_blocks = [16, 32, 64, 128]
    n_blocks = [16, 32, 64, 128, 256]
    k_blocks = [32, 64, 128]
    g_blocks = [1, 4, 8, 16, 32]
    ksplits = [1, 2]

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


def filter_occupancy(candidate: Dict[str, Any], is_moe: bool) -> bool:
    m = candidate["BLOCK_SIZE_M"]
    n = candidate["BLOCK_SIZE_N"]
    k = candidate["BLOCK_SIZE_K"]
    w = candidate.get("num_warps", 4)

    block_vol = m * n * k
    if block_vol > 128 * 256 * 128:
        return False

    if m * n < 1024 and w > 4:
        return False

    if is_moe:
        stages = candidate.get("num_stages", 2)
        if stages >= 4 and block_vol > 64 * 128 * 64:
            return False

    return True


class DenseCandidateBuilder:
    def __init__(self, is_amd: bool, is_fp8: bool, backend: str):
        self.is_amd = is_amd
        self.is_fp8 = is_fp8
        self.backend = backend

    def build(self) -> List[Dict[str, Any]]:
        # Separate backend-specific candidate spaces
        if self.backend == "aiter_triton":
            if not self.is_fp8:
                raise RuntimeError("aiter_triton dense builder is currently implemented for fp8 only")
            base = get_base_candidates_aiter_triton_dense_fp8()
            results = [c for c in base if filter_occupancy(c, is_moe=False)]
            dedup = {frozenset(d.items()): d for d in results}
            return list(dedup.values())

        # Triton path
        base = get_base_candidates_triton_dense()
        results = []

        for p in base:
            if self.is_amd and self.is_fp8:
                for kpack in [1, 2]:
                    for nonkdim in [16, 32]:
                        c = p.copy()
                        c["kpack"] = kpack
                        c["matrix_instr_nonkdim"] = nonkdim
                        if filter_occupancy(c, is_moe=False):
                            results.append(c)
            else:
                if filter_occupancy(p, is_moe=False):
                    results.append(p)

        dedup = {frozenset(d.items()): d for d in results}
        return list(dedup.values())


class MoECandidateBuilder:
    def __init__(self, backend: str):
        self.backend = backend

    def build(self) -> List[Dict[str, Any]]:
        if self.backend == "aiter_triton":
            # Honest fail-closed for now. AITER public MoE wrapper computes its own config surface.
            raise RuntimeError("aiter_triton MoE tuning is not implemented yet")

        base = get_base_candidates_triton_dense()
        results = []

        stages_options = [2, 3, 4, 5]
        for p in base:
            for stages in stages_options:
                c = p.copy()
                c["num_stages"] = stages
                if filter_occupancy(c, is_moe=True):
                    results.append(c)

        dedup = {frozenset(d.items()): d for d in results}
        return list(dedup.values())


def mutate_candidate(candidate: Dict[str, Any], is_moe: bool, backend: str) -> List[Dict[str, Any]]:
    mutations = []

    # Triton-specific mutation surface
    if backend == "triton":
        for w in [2, 4, 8]:
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
            for adj in [gs // 2, gs * 2]:
                if 1 <= adj <= 64 and adj != gs:
                    c = candidate.copy()
                    c["GROUP_SIZE_M"] = adj
                    mutations.append(c)

    elif backend == "aiter_triton":
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
            for adj in [gs // 2, gs * 2]:
                if 1 <= adj <= 64 and adj != gs:
                    c = candidate.copy()
                    c["GROUP_SIZE_M"] = adj
                    mutations.append(c)

        if "NUM_KSPLIT" in candidate:
            ks = candidate["NUM_KSPLIT"]
            for adj in [1, 2, 4]:
                if adj != ks:
                    c = candidate.copy()
                    c["NUM_KSPLIT"] = adj
                    mutations.append(c)

    valid_mutations = [m for m in mutations if filter_occupancy(m, is_moe)]
    dedup = {frozenset(d.items()): d for d in valid_mutations}
    return list(dedup.values())