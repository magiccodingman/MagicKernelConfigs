import itertools
from typing import List, Dict, Any

def get_base_candidates() -> List[Dict[str, Any]]:
    """
    Defines a reasonable high-fidelity space of generic block shapes.
    Exhaustive search is used, taking cross-products of reasonable powers of 2.
    """
    m_blocks = [16, 32, 64, 128]
    n_blocks = [16, 32, 64, 128, 256]
    k_blocks = [32, 64, 128, 256]
    g_blocks = [1, 8, 16, 32]
    warps = [2, 4, 8]
    
    candidates = []
    for m, n, k, g, w in itertools.product(m_blocks, n_blocks, k_blocks, g_blocks, warps):
        candidates.append({
            "BLOCK_SIZE_M": m,
            "BLOCK_SIZE_N": n,
            "BLOCK_SIZE_K": k,
            "GROUP_SIZE_M": g,
            "num_warps": w
        })
    return candidates

def filter_occupancy(candidate: Dict[str, Any], is_moe: bool) -> bool:
    """
    Pre-runtime screening for invalid combinations likely to fail or perform terribly.
    - shared memory checks (heuristic bounds)
    - incompatible thread counts versus matrix dimensions
    """
    m = candidate["BLOCK_SIZE_M"]
    n = candidate["BLOCK_SIZE_N"]
    k = candidate["BLOCK_SIZE_K"]
    w = candidate["num_warps"]
    
    # 1. Block Volume (proxy for shared memory / registers)
    # A block volume that is too high practically guarantees local memory exhaustion on AMD/NV
    block_vol = m * n * k
    if block_vol > 128 * 256 * 128:
        return False
        
    # 2. Warp waste checking (extremely thin tiles using maximum warps just causes contention)
    if m * n < 1024 and w > 4:
        return False
        
    # 3. MoE specific checks (num_stages implies software pipelining buffer sizes)
    if is_moe:
        stages = candidate.get("num_stages", 2)
        if stages >= 4 and block_vol > 64 * 128 * 64:
            return False
            
    return True

class DenseCandidateBuilder:
    def __init__(self, is_amd: bool, is_fp8: bool):
        self.is_amd = is_amd
        self.is_fp8 = is_fp8
        
    def build(self) -> List[Dict[str, Any]]:
        base = get_base_candidates()
        results = []
        
        for p in base:
            # Expand with AMD-specific dense parameters if applicable
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
                    
        # Apply deduplication
        dedup = {frozenset(d.items()): d for d in results}
        return list(dedup.values())
        
class MoECandidateBuilder:
    def __init__(self):
        pass
        
    def build(self) -> List[Dict[str, Any]]:
        base = get_base_candidates()
        results = []
        
        stages_options = [2, 3, 4, 5]
        for p in base:
            for stages in stages_options:
                c = p.copy()
                c["num_stages"] = stages
                if filter_occupancy(c, is_moe=True):
                    results.append(c)
                    
        # Apply deduplication
        dedup = {frozenset(d.items()): d for d in results}
        return list(dedup.values())

def mutate_candidate(candidate: Dict[str, Any], is_moe: bool) -> List[Dict[str, Any]]:
    """
    Controlled exploration of neighboring kernel configurations.
    Returns structurally similar but varied mappings off of a winning candidate.
    """
    mutations = []
    
    # Toggle warps
    for w in [2, 4, 8]:
        if w != candidate.get("num_warps", -1):
            c = candidate.copy()
            c["num_warps"] = w
            mutations.append(c)
            
    # Nudges to primary block sizes (up and down by 2x)
    for dim in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]:
        if dim in candidate:
            current = candidate[dim]
            for adj in [current // 2, current * 2]:
                if 16 <= adj <= 256 and adj != current:
                    c = candidate.copy()
                    c[dim] = adj
                    mutations.append(c)
                    
    # Nudge group sizes
    if "GROUP_SIZE_M" in candidate:
        gs = candidate["GROUP_SIZE_M"]
        for adj in [gs // 2, gs * 2]:
            if 1 <= adj <= 64 and adj != gs:
                c = candidate.copy()
                c["GROUP_SIZE_M"] = adj
                mutations.append(c)

    # Filter out mutations that fail the occupancy rules or look mathematically invalid
    valid_mutations = []
    for m in mutations:
        if filter_occupancy(m, is_moe):
            valid_mutations.append(m)
            
    # Deduplicate before returning
    dedup = {frozenset(d.items()): d for d in valid_mutations}
    return list(dedup.values())
