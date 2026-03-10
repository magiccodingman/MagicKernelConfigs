import json
from pathlib import Path
from typing import Dict, Any

# Using the atomic_write_json helper from the filesystem module
from Utils.filesystem import atomic_write_json

def write_batch_keyed_json(bucket_winners: Dict[int, Dict[str, Any]], output_path: Path, is_moe: bool):
    """
    Writes the tuned kernel parameters to a JSON file keyed by batch size strings.
    Strips invalid schema elements so vLLM's existing loaders accept the output securely.
    """
    final_dict = {}
    
    # Allowed schema fields according to vLLM's exact acceptance specs
    # Note: kpack and matrix_instr_nonkdim are strictly Dense ROCm only for FP8.
    dense_allowed = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "kpack", "matrix_instr_nonkdim"}
    moe_allowed = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
    
    allowed = moe_allowed if is_moe else dense_allowed
    
    for m_val, candidate in bucket_winners.items():
        # Only copy keys that are valid for this specific schema
        clean_candidate = {k: v for k, v in candidate.items() if k in allowed}
        final_dict[str(m_val)] = clean_candidate
        
    atomic_write_json(output_path, final_dict)

def write_manifest(manifest_path: Path, run_metadata: Dict[str, Any]):
    """
    Writes the run metadata and any raised warnings to manifest.json 
    to track the exact hardware/tensor-parallel matrix that produced these outputs.
    """
    atomic_write_json(manifest_path, run_metadata)
