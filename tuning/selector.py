from typing import Dict, Any, List

def calculate_neighbor_penalty(candidate_profiles: Dict[int, Dict[str, float]], target_m: int) -> float:
    """
    Penalizes candidates that perform terribly in neighboring batch regions.
    A candidate that is extremely fast at M=64 but 5x slower at M=16 or M=256
    is unstable and likely hitting some catastrophic occupancy cliff.
    """
    sorted_ms = sorted(list(candidate_profiles.keys()))
    if target_m not in sorted_ms:
        return 0.0
        
    idx = sorted_ms.index(target_m)
    neighbors = []
    if idx > 0:
        neighbors.append(sorted_ms[idx - 1])
    if idx < len(sorted_ms) - 1:
        neighbors.append(sorted_ms[idx + 1])
        
    # We want to encourage smooth scaling.
    # If the local time scales much worse than O(N) relative to its neighbors, we add a penalty
    penalty = 0.0
    target_metric = candidate_profiles[target_m]["local_ms"]
    
    for n_m in neighbors:
        n_metric = candidate_profiles[n_m]["local_ms"]
        if n_metric == float('inf') or target_metric == float('inf'):
            return float('inf')
            
        ratio = (n_metric / n_m) / (target_metric / target_m)
        if ratio > 5.0 or ratio < 0.2:
            penalty += target_metric * 0.5  # Heavy 50% penalty for massive unstability
            
    return penalty

class CalibrationSelectionError(Exception):
    """Raised when kernel configuration selection yields no viable candidates."""
    pass

def fallback_select_best_candidate(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Deterministic fallback logic. 
    Selects the absolute best candidate for *each evaluated bucket* based purely on the lowest local_ms (concurrency=1).
    Ignores neighbor penalties and parallel metrics, guaranteeing an output for every profiled batch size.
    """
    if not candidate_results:
        return {}
        
    eval_buckets = sorted({
        bucket
        for result in candidate_results
        for bucket in result["profiles"].keys()
    })
    winners = {}
    
    for m in eval_buckets:
        best_score = float('inf')
        best_candidate = None
        
        for result in candidate_results:
            profiles = result["profiles"]
            if m not in profiles:
                continue
                
            local_ms = profiles[m]["local_ms"]
            
            if local_ms < best_score:
                best_score = local_ms
                best_candidate = result["candidate"]
                
        if best_candidate is not None:
            winners[m] = best_candidate
            
    return winners

def score_and_select_winners(candidate_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Selects the winning kernel configuration for each evaluated batch bucket.
    candidate_results format:
    [
        {
            "candidate": {"BLOCK_SIZE_M": 64, ...},
            "profiles": {
                16: {"local_ms": 1.2, "parallel_ms": 1.4},
                64: {"local_ms": 2.1, "parallel_ms": 2.5},
            }
        }, ...
    ]
    Returns mapping from M -> winning candidate dict.
    """
    if not candidate_results:
        return {}
        
    # Determine the M buckets evaluated accurately handling partial failures
    eval_buckets = sorted({
        bucket
        for result in candidate_results
        for bucket in result["profiles"].keys()
    })
    winners = {}
    
    for m in eval_buckets:
        best_score = float('inf')
        best_candidate = None
        
        for result in candidate_results:
            profiles = result["profiles"]
            if m not in profiles:
                continue
                
            local_ms = profiles[m]["local_ms"]
            parallel_ms = profiles[m]["parallel_ms"]
            
            if local_ms == float('inf') or parallel_ms == float('inf'):
                continue
                
            # Baseline score is heavily weighted towards parallel contention realism
            # since vLLM runs under high TP pressure in production.
            combined_score = (local_ms * 0.3) + (parallel_ms * 0.7)
            
            # Apply neighbor stability penalty
            stability_penalty = calculate_neighbor_penalty(profiles, m)
            combined_score += stability_penalty
            
            if combined_score < best_score:
                best_score = combined_score
                best_candidate = result["candidate"]
                
        if best_candidate is not None:
            winners[m] = best_candidate
            
    return winners
