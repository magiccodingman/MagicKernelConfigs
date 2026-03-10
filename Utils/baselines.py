import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

def get_default_baseline_path() -> Path:
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    else:
        base = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
        
    return Path(base) / "MagicKernelConfigs" / "Baselines"

def get_baseline_file_path(baseline_path_arg: str, gpu_name: str, gfx_version: str, backend: str, dtype_family: str, dtype_subtype: str) -> Path:
    if baseline_path_arg:
        base = Path(baseline_path_arg)
    else:
        base = get_default_baseline_path()
        
    normalized_gpu = gpu_name.replace(" ", "_")
    
    path = base / normalized_gpu
    if gfx_version:
        path = path / gfx_version
        
    path = path / backend / dtype_family / dtype_subtype / "baseline.json"
    return path

class BaselineCache:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()
        
    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load baseline cache at {self.path}: {e}")
                
        return {
            "candidate_builder_version": 1,
            "kernel_harness_version": 1,
            "schema_version": 1,
            "gpu": None,
            "gfx": None,
            "backend": None,
            "dtype_family": None,
            "dtype_subtype": None,
            "block_n": None,
            "block_k": None,
            "shapes": {}  # Maps "N_K_is_moe" to {"completed": 0, "total": 0, "results": {}}
        }
        
    def init_metadata(self, gpu: str, gfx: str, backend: str, dtype_family: str, dtype_subtype: str, block_n: int, block_k: int):
        self.data["gpu"] = gpu
        self.data["gfx"] = gfx
        self.data["backend"] = backend
        self.data["dtype_family"] = dtype_family
        self.data["dtype_subtype"] = dtype_subtype
        self.data["block_n"] = block_n
        self.data["block_k"] = block_k
        self._save()
        
    def is_compatible(self, gpu: str, gfx: str, backend: str, dtype_family: str, dtype_subtype: str) -> bool:
        if self.data["candidate_builder_version"] != 1: return False
        if self.data["kernel_harness_version"] != 1: return False
        if self.data["schema_version"] != 1: return False
        if self.data.get("gpu") and self.data["gpu"] != gpu: return False
        if self.data.get("gfx") and self.data["gfx"] != gfx: return False
        if self.data.get("backend") and self.data["backend"] != backend: return False
        if self.data.get("dtype_family") and self.data["dtype_family"] != dtype_family: return False
        if self.data.get("dtype_subtype") and self.data["dtype_subtype"] != dtype_subtype: return False
        return True
        
    def clear(self):
        self.data = {
            "candidate_builder_version": 1,
            "kernel_harness_version": 1,
            "schema_version": 1,
            "gpu": None,
            "gfx": None,
            "backend": None,
            "dtype_family": None,
            "dtype_subtype": None,
            "block_n": None,
            "block_k": None,
            "shapes": {}
        }
        self._save()
        
    def get_shape_key(self, n: int, k: int, is_moe: bool) -> str:
        return f"{n}_{k}_{'moe' if is_moe else 'dense'}"
        
    def setup_shape(self, n: int, k: int, is_moe: bool, total_candidates: int) -> list:
        """Returns previously evaluated results for this shape, and ensures the structure exists."""
        key = self.get_shape_key(n, k, is_moe)
        if key not in self.data["shapes"]:
            self.data["shapes"][key] = {
                "completed": 0,
                "total": total_candidates,
                "results": {}
            }
            self._save()
            return []
            
        # If total changed, update it
        self.data["shapes"][key]["total"] = total_candidates
        return list(self.data["shapes"][key]["results"].values())
        
    def has_candidate(self, n: int, k: int, is_moe: bool, candidate_key: str) -> bool:
        key = self.get_shape_key(n, k, is_moe)
        if key in self.data["shapes"]:
            return candidate_key in self.data["shapes"][key]["results"]
        return False
        
    def add_result(self, n: int, k: int, is_moe: bool, candidate_key: str, candidate_result: Dict[str, Any]):
        key = self.get_shape_key(n, k, is_moe)
        if key not in self.data["shapes"]:
            self.setup_shape(n, k, is_moe, 0)
            
        if candidate_key not in self.data["shapes"][key]["results"]:
            self.data["shapes"][key]["results"][candidate_key] = candidate_result
            self.data["shapes"][key]["completed"] = len(self.data["shapes"][key]["results"])
            self._save()
            
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp_base_", suffix=".json")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            os.replace(tmp_path, self.path)
        except Exception as e:
            print(f"Warning: Failed to save baseline cache atomically: {e}", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
