from pathlib import Path
from collections import defaultdict

# Importing the shared shape discovery utilities
from Utils.vllm_config_utils import (
    extract_runtime_gemm_shapes,
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

def generate_inventory(model_path: dict, tp_target: int, device_name: str, 
                       block_n: int, block_k: int, dtype_label: str) -> list[ConfigInventory]:
    """
    Generates the strict target inventory (filenames and parameters) based ONLY on 
    model shape discovery. 
    Returns the required config specifications.
    
    The device_name here must be the exact runtime device name so it matches vLLM exactly.
    """
    # ------------------------------------------------------------
    # 1. Discover shapes across TP levels
    # ------------------------------------------------------------
    tp_levels = []
    v = 1
    while v <= tp_target:
        tp_levels.append(v)
        v *= 2

    shapes_by_tp = {}
    shaped_by_tp = {}

    for tp in tp_levels:
        shaped_tp = extract_runtime_gemm_shapes(model_path, tp)
        shaped_by_tp[tp] = shaped_tp
        shapes_by_tp[tp] = {(N, K) for N, K, _ in shaped_tp}

    # ------------------------------------------------------------
    # 2. Determine shapes unique to the target TP level
    # ------------------------------------------------------------
    current_shapes = shapes_by_tp[tp_target]

    if tp_target == 1:
        target_shapes = current_shapes
    else:
        prev_tp = tp_target // 2
        prev_shapes = shapes_by_tp.get(prev_tp, set())
        target_shapes = current_shapes - prev_shapes

    # ------------------------------------------------------------
    # 3. Filter final shaped list to only those shapes
    # ------------------------------------------------------------
    shaped = [
        (N, K, label)
        for (N, K, label) in shaped_by_tp[tp_target]
        if (N, K) in target_shapes
    ]

    print("\n=== shaped GEMM Shapes Discovered ===")

    for shape in shaped:
        print(shape)

    print("==================================\n")
    
    
    unique_by_shape: dict[tuple[int, int], set[str]] = defaultdict(set)
    for N, K, label in shaped:
        if should_keep_shape(label):
            unique_by_shape[(N, K)].add(label)

    print("\n=== Shapes To Keep ===")

    for uShape in unique_by_shape:
        print(uShape)

    print("==================================\n")
            
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
        

    print("\n=== Inventory To Keep ===")

    for uShape in inventory:
        print(uShape.filename)

    print("==================================\n")

    validate_expected_shapes(inventory)

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

def validate_expected_shapes(inventory: list[ConfigInventory]) -> None:
    """
    Validates that the inventory contains the expected GEMM shapes and reports:
    - Found shapes
    - Missing shapes
    - Extra shapes
    """

    expected_shapes = {
        (8192, 5120),
        (5120, 3072),
        (17408, 5120),
        (5120, 8704),
        (7168, 5120),
    }

    found_shapes = {(item.n, item.k) for item in inventory}

    missing = expected_shapes - found_shapes
    extra = found_shapes - expected_shapes
    present = expected_shapes & found_shapes

    print("\n=== Shape Validation ===")

    print("\n✔ Found Expected Shapes:")
    if present:
        for n, k in sorted(present):
            print(f"  N={n}, K={k}")
    else:
        print("  None")

    print("\n❌ Missing Expected Shapes:")
    if missing:
        for n, k in sorted(missing):
            print(f"  N={n}, K={k}")
    else:
        print("  None")

    print("\n⚠ Extra Shapes Discovered:")
    if extra:
        for n, k in sorted(extra):
            print(f"  N={n}, K={k}")
    else:
        print("  None")

    print("\nSummary:")
    print(f"  Expected: {len(expected_shapes)}")
    print(f"  Found: {len(found_shapes)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Extra: {len(extra)}")

    print("=========================\n")