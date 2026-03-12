import json
from pathlib import Path
from typing import Dict, Any

from Utils.filesystem import atomic_write_json


def _sanitize_dense_vllm_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist ONLY the vLLM-facing dense FP8 config surface.

    Internal-only tuning knobs must never leak here unless your target vLLM kernel
    truly consumes them.
    """
    out = {
        "BLOCK_SIZE_M": int(candidate["BLOCK_SIZE_M"]),
        "BLOCK_SIZE_N": int(candidate["BLOCK_SIZE_N"]),
        "BLOCK_SIZE_K": int(candidate["BLOCK_SIZE_K"]),
        "GROUP_SIZE_M": int(candidate.get("GROUP_SIZE_M", 1)),
        "num_warps": int(candidate.get("num_warps", 4)),
        "num_stages": int(candidate.get("num_stages", 2)),
    }

    # Keep these only if your local vLLM kernel actually consumes them.
    if "kpack" in candidate:
        out["kpack"] = int(candidate["kpack"])
    if "matrix_instr_nonkdim" in candidate:
        out["matrix_instr_nonkdim"] = int(candidate["matrix_instr_nonkdim"])

    return out


def _sanitize_moe_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "BLOCK_SIZE_M": int(candidate["BLOCK_SIZE_M"]),
        "BLOCK_SIZE_N": int(candidate["BLOCK_SIZE_N"]),
        "BLOCK_SIZE_K": int(candidate["BLOCK_SIZE_K"]),
    }

    if "GROUP_SIZE_M" in candidate:
        out["GROUP_SIZE_M"] = int(candidate["GROUP_SIZE_M"])
    if "num_warps" in candidate:
        out["num_warps"] = int(candidate["num_warps"])
    if "num_stages" in candidate:
        out["num_stages"] = int(candidate["num_stages"])

    if "kpack" in candidate:
        out["kpack"] = int(candidate["kpack"])
    if "matrix_instr_nonkdim" in candidate:
        out["matrix_instr_nonkdim"] = int(candidate["matrix_instr_nonkdim"])

    return out


def write_batch_keyed_json(bucket_winners, output_path, is_moe, backend):

    final_dict = {}

    for m_val, candidate in bucket_winners.items():

        if backend == "aiter_triton":

            # Persist Triton-compatible schema
            clean = {
                "BLOCK_SIZE_M": candidate["BLOCK_SIZE_M"],
                "BLOCK_SIZE_N": candidate["BLOCK_SIZE_N"],
                "BLOCK_SIZE_K": candidate["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": candidate.get("GROUP_SIZE_M", 1),

                # safe defaults since we didn't tune them
                "num_warps": 4,
                "num_stages": 2,
            }

        else:  # pure triton

            clean = {
                "BLOCK_SIZE_M": candidate["BLOCK_SIZE_M"],
                "BLOCK_SIZE_N": candidate["BLOCK_SIZE_N"],
                "BLOCK_SIZE_K": candidate["BLOCK_SIZE_K"],
                "GROUP_SIZE_M": candidate.get("GROUP_SIZE_M", 1),
                "num_warps": candidate.get("num_warps", 4),
                "num_stages": candidate.get("num_stages", 2),
            }

            if "kpack" in candidate:
                clean["kpack"] = candidate["kpack"]

            if "matrix_instr_nonkdim" in candidate:
                clean["matrix_instr_nonkdim"] = candidate["matrix_instr_nonkdim"]

        final_dict[str(m_val)] = clean

    atomic_write_json(output_path, final_dict)


def write_manifest(manifest_path: Path, run_metadata: Dict[str, Any]):
    atomic_write_json(manifest_path, run_metadata)