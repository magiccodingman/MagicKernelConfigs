import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

def get_base_output_dir(model_path: Path, vendor: str, exact_device_name: str, 
                        gfx_version: str, backend: str, dtype_family: str, tp_max: int) -> Path:
    """
    Returns the target output directory for this calibration run.
    Uses the normalized GPU name (replacing spaces with underscores).
    If a valid gfx_version is provided (i.e. for AMD), it is included in the path.
    """
    normalized_gpu_name = exact_device_name.replace(" ", "_")
    
    base_dir = model_path / "MagicKernelConfigs" / vendor / normalized_gpu_name
    
    if gfx_version:
        base_dir = base_dir / gfx_version
        
    base_dir = base_dir / backend / dtype_family / f"tp_{tp_max}"
    
    return base_dir

def setup_output_directories(base_dir: Path):
    """
    Ensures that the output directory and its dense/moe subdirectories exist.
    """
    dense_dir = base_dir / "dense"
    moe_dir = base_dir / "moe"
    
    dense_dir.mkdir(parents=True, exist_ok=True)
    moe_dir.mkdir(parents=True, exist_ok=True)
    
    return dense_dir, moe_dir

def should_write_file(filepath: Path) -> bool:
    """
    Determines whether a file should be written based on overwrite/skip logic.
    - If file doesn't exist, return True.
    - If file exists and is 0 bytes, return True.
    - If file exists and >0 bytes, prompt the user.
    """
    if not filepath.exists():
        return True
        
    if filepath.stat().st_size == 0:
        return True
        
    while True:
        try:
            choice = input(f"\n⚠️ File {filepath} already exists and is non-empty.\nOverwrite? [y/N]: ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            if choice in ['n', 'no', '']:
                return False
        except (EOFError, KeyboardInterrupt):
            print("\nAborting.")
            sys.exit(1)

def atomic_write_json(filepath: Path, data: dict):
    """
    Writes a JSON file atomically using a temporary file.
    """
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    # atomic replace
    os.replace(tmp_path, filepath)

def write_manifest(filepath: Path, manifest_data: dict):
    """
    Writes the manifest.json atomically. Overwrites by default if we are running.
    """
    atomic_write_json(filepath, manifest_data)
