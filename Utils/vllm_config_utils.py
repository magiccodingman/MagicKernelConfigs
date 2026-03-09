import importlib.util
import json
import sys
from pathlib import Path

# Extrapolated naming conventions for the different precisions.
# You can tweak these strings if your specific vLLM fork names them slightly differently.
DTYPE_CONFIGS = {
    "fp8": {
        "utils_file": "fp8_utils.py",
        "bench_script": "benchmark_fp8_w8a8_gemm.py",
        "file_label": "fp8_w8a8"
    },
    "fp4": {
        "utils_file": "fp4_utils.py",
        "bench_script": "benchmark_fp4_w4a4_gemm.py",
        "file_label": "fp4_w4a4"
    },
    "int8": {
        "utils_file": "int8_utils.py",
        "bench_script": "benchmark_int8_w8a8_gemm.py",
        "file_label": "int8_w8a8"
    },
    "int4": {
        "utils_file": "int4_utils.py",
        "bench_script": "benchmark_int4_w4a4_gemm.py",
        "file_label": "int4_w4a4"
    }
}


def find_vllm_base_path() -> Path:
    try:
        import vllm
        return Path(vllm.__path__[0])
    except ImportError:
        print("Error: vLLM is not installed in the current Python environment.")
        sys.exit(1)


def load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {path}: {e}")
        sys.exit(1)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def try_get_device_name(vllm_base: Path, utils_file: str) -> str:
    utils_path = next(vllm_base.rglob(utils_file), None)
    if not utils_path:
        return "AMD_Radeon_Graphics"

    try:
        spec = importlib.util.spec_from_file_location("v_utils", utils_path)
        if spec is None or spec.loader is None:
            return "AMD_Radeon_Graphics"

        mod = importlib.util.module_from_spec(spec)
        sys.modules["v_utils"] = mod
        spec.loader.exec_module(mod)
        return mod.current_platform.get_device_name().replace(" ", "_")
    except Exception:
        return "AMD_Radeon_Graphics"


def get_dtype_paths(vllm_base: Path, utils_file: str, bench_script_name: str) -> tuple[Path, Path]:
    utils_path = next(vllm_base.rglob(utils_file), None)
    if utils_path is None:
        print(f"Error: Could not locate {utils_file} inside vLLM.")
        sys.exit(1)

    configs_dir = utils_path.parent / "configs"
    bench = vllm_base.parent / "benchmarks" / bench_script_name
    return configs_dir, bench


def normalize_text_config(root_config: dict) -> dict:
    return root_config.get("text_config", root_config)


def build_tp_levels(tp_max: int) -> list[int]:
    levels = []
    value = 1
    while value <= tp_max:
        levels.append(value)
        value *= 2
    return levels


def should_keep_shape(N: int, K: int, block_n: int, block_k: int) -> bool:
    return N >= block_n and K >= block_k


def make_filename(N: int, K: int, device_name: str, block_n: int, block_k: int, dtype_label: str) -> str:
    return (
        f"N={N},K={K},device_name={device_name},"
        f"dtype={dtype_label},block_shape=[{block_n},{block_k}].json"
    )


def extract_qwen35_runtime_text_shapes(root_config: dict) -> list[tuple[str, int, int, str]]:
    """
    Returns runtime-oriented GEMM shapes for Qwen3.5 text path.

    Output:
        (shard_type, N, K, label)
    """
    text_cfg = normalize_text_config(root_config)

    hidden_size = text_cfg.get("hidden_size")
    intermediate_size = text_cfg.get("intermediate_size")
    num_heads = text_cfg.get("num_attention_heads")
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim")

    if not hidden_size:
        return []

    shapes = []

    # ---- QKV merged ----
    if num_heads and num_kv_heads and head_dim:
        qkv_out = hidden_size + 2 * (num_kv_heads * head_dim)
        shapes.append(("column", qkv_out, hidden_size, "text_qkv_merged"))

    # ---- MLP ----
    if intermediate_size:
        shapes.append(("column", intermediate_size, hidden_size, "text_mlp_expand"))
        shapes.append(("row", hidden_size, intermediate_size, "text_mlp_down"))

    # ---- Qwen3.5 text-path extras ----
    model_type = text_cfg.get("model_type", "")
    if model_type == "qwen3_5_text":
        shapes.append(("column", 8192, hidden_size, "text_qwen35_linear_attn_expand"))
        shapes.append(("row", hidden_size, 3072, "text_qwen35_linear_attn_down"))

    return shapes


def apply_tp_sharding(
    raw_shapes: list[tuple[str, int, int, str]],
    tp: int,
) -> list[tuple[int, int, str]]:
    """
    column-parallel: shard N
    row-parallel: shard K
    """
    result = []

    for shard_type, N, K, label in raw_shapes:
        if shard_type == "column":
            result.append((ceil_div(N, tp), K, label))
        elif shard_type == "row":
            result.append((N, ceil_div(K, tp), label))

    return result