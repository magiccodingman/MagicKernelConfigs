from pathlib import Path
from collections import defaultdict

# Importing the shared shape discovery utilities
from Utils.vllm_config_utils import (
    extract_qwen35_runtime_text_shapes,
    apply_tp_sharding,
    should_keep_shape,
    make_filename
)
from Utils.filesystem import should_write_file

class ConfigInventory:
    def __init__(self, n: int, k: int, filename: str, labels: list[str], is_moe: bool):
        self.n = n
        self.k = k
        self.filename = filename
        self.labels = labels
        self.is_moe = is_moe
        self.is_skipped = False
        self.output_path = None

def is_moe_label(label: str) -> bool:
    """Helper to detect if a module is MoE related based on shape labels."""
    return "moe" in label.lower()

def generate_inventory(root_config: dict, tp_target: int, device_name: str, 
                       block_n: int, block_k: int, dtype_label: str) -> list[ConfigInventory]:
    """
    Generates the strict target inventory (filenames and parameters) based ONLY on 
    model shape discovery. 
    Returns the required config specifications.
    
    The device_name here must be the exact runtime device name so it matches vLLM exactly.
    """
    # 1. Shape discovery directly from model
    raw_shapes = extract_qwen35_runtime_text_shapes(root_config)
    
    # 2. TP sharding for the exact requested target
    shaped = apply_tp_sharding(raw_shapes, tp_target)
    
    unique_by_shape: dict[tuple[int, int], set[str]] = defaultdict(set)
    for N, K, label in shaped:
        if should_keep_shape(N, K, block_n, block_k):
            unique_by_shape[(N, K)].add(label)
            
    inventory = []
    
    for (N, K), labels in sorted(unique_by_shape.items()):
        # Generates filename with exact device text
        fname = make_filename(N, K, device_name, block_n, block_k, dtype_label)
        has_moe = any(is_moe_label(lbl) for lbl in labels)
        
        item = ConfigInventory(
            n=N,
            k=K,
            filename=fname,
            labels=sorted(labels),
            is_moe=has_moe
        )
        inventory.append(item)
        
    return inventory

def resolve_inventory_paths(inventory: list[ConfigInventory], dense_dir: Path, moe_dir: Path) -> list[ConfigInventory]:
    """
    Assigns final output paths to inventory items based on whether they are Dense or MoE,
    and applies skip/overwrite user prompts to prune skipped targets.
    Returns the active inventory to be generated.
    """
    active_inventory = []
    
    for item in inventory:
        target_dir = moe_dir if item.is_moe else dense_dir
        item.output_path = target_dir / item.filename
        
        if should_write_file(item.output_path):
            active_inventory.append(item)
        else:
            item.is_skipped = True
            print(f"Skipping {item.filename} (already exists).")
            
    return active_inventory
