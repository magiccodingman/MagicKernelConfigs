import importlib.util
import json
import sys
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
import torch.nn as nn
from collections import OrderedDict
import fnmatch

try:
    from transformers.configuration_utils import PreTrainedConfig
except Exception:
    PreTrainedConfig = object

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


def try_get_device_name(vllm_base: Path, utils_file: str, device_override: str = None) -> str:
    if device_override:
        return device_override.replace(" ", "_")

    try:
        from vllm.platforms import current_platform
        return current_platform.get_device_name().replace(" ", "_")
    except Exception:
        print("Warning: Dynamic device detection failed. Defaulting to 'Unknown_Device'.")
        return "Unknown_Device"


def get_dtype_paths(vllm_base: Path, utils_file: str, bench_script_name: str) -> tuple[Path, Path]:
    utils_path = next(vllm_base.rglob(utils_file), None)
    if utils_path is None:
        print(f"Error: Could not locate {utils_file} inside vLLM.")
        sys.exit(1)

    configs_dir = utils_path.parent / "configs"
    bench = vllm_base.parent / "benchmarks" / bench_script_name
    return configs_dir, bench

def build_tp_levels(tp_max: int) -> list[int]:
    levels = []
    value = 1
    while value <= tp_max:
        levels.append(value)
        value *= 2
    return levels


def should_keep_shape(label: str) -> bool:
    """
    Keep only shapes from runtime-relevant fused roles.
    Works best when labels are normalized into semantic roles.
    """

    allowed_roles = (
        "text_qkv_merged",
        "text_mlp_expand",
        "text_mlp_down",
        "text_qwen35_linear_attn_expand",
        "text_qwen35_linear_attn_down",
    )

    return any(role in label for role in allowed_roles)

def make_filename(N: int, K: int, device_name: str, block_n: int, block_k: int, dtype_label: str) -> str:
    return (
        f"N={N},K={K},device_name={device_name},"
        f"dtype={dtype_label},block_shape=[{block_n},{block_k}].json"
    )

TP_COLWISE = {"colwise", "packed_colwise"}
TP_ROWWISE = {"rowwise", "packed_rowwise", "local_packed_rowwise"}

# Fallback only when a config has no TP plan at all.
FALLBACK_COLWISE_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "qkv_proj", "gate_up_proj"
}
FALLBACK_ROWWISE_SUFFIXES = {
    "o_proj", "down_proj", "out_proj"
}


def _tp_levels(tp_max: int) -> list[int]:
    levels = []
    v = 1
    while v <= max(1, int(tp_max)):
        levels.append(v)
        v *= 2
    return levels


def _iter_subconfigs(cfg, seen=None):
    if seen is None:
        seen = set()

    if cfg is None or id(cfg) in seen:
        return

    seen.add(id(cfg))
    yield cfg

    # Prefer base_config_key first if present.
    base_key = getattr(cfg, "base_config_key", None)
    if base_key and hasattr(cfg, base_key):
        sub = getattr(cfg, base_key)
        if isinstance(sub, PreTrainedConfig):
            yield from _iter_subconfigs(sub, seen)

    # Then recurse all nested config-like attrs.
    for name, value in vars(cfg).items():
        if name.startswith("_"):
            continue
        if isinstance(value, PreTrainedConfig):
            yield from _iter_subconfigs(value, seen)


def _pick_tp_config(root_cfg):
    """
    Pick the most relevant config object for TP metadata.
    Preference:
      1) a config with base_model_tp_plan
      2) if multiple, prefer one whose model_type contains 'text'
      3) otherwise first one found
    """
    all_cfgs = list(_iter_subconfigs(root_cfg))
    with_tp = [c for c in all_cfgs if getattr(c, "base_model_tp_plan", None)]

    if not with_tp:
        return root_cfg

    with_tp.sort(
        key=lambda c: (
            0 if "text" in str(getattr(c, "model_type", "")).lower() else 1,
            0 if getattr(root_cfg, "base_config_key", None)
                 and hasattr(root_cfg, root_cfg.base_config_key)
                 and c is getattr(root_cfg, root_cfg.base_config_key)
              else 1,
        )
    )
    return with_tp[0]


def _normalized_names(name: str) -> list[str]:
    """
    Try a few normalized forms so TP plan patterns like 'layers.*.self_attn.q_proj'
    can match names such as 'model.layers.0.self_attn.q_proj'.
    """
    names = [name]
    if ".layers." in name:
        names.append("layers." + name.split(".layers.", 1)[1])
    if ".model.layers." in name:
        names.append("layers." + name.split(".model.layers.", 1)[1])
    return list(dict.fromkeys(names))


def _tp_kind_for_name(name: str, tp_plan: dict[str, str] | None) -> str | None:
    if not tp_plan:
        return None

    candidates = _normalized_names(name)
    for pattern, kind in tp_plan.items():
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return kind
    return None


def _fallback_tp_kind(name: str) -> str | None:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in FALLBACK_COLWISE_SUFFIXES:
        return "colwise"
    if leaf in FALLBACK_ROWWISE_SUFFIXES:
        return "rowwise"
    return None


def _linear_shape(module) -> tuple[int, int] | None:
    # Standard nn.Linear and most wrappers
    if hasattr(module, "out_features") and hasattr(module, "in_features"):
        try:
            return int(module.out_features), int(module.in_features)
        except Exception:
            pass

    # Fallback to a real 2D weight
    weight = getattr(module, "weight", None)
    if weight is not None:
        shape = getattr(weight, "shape", None)
        if shape is not None and len(shape) == 2:
            try:
                return int(shape[0]), int(shape[1])
            except Exception:
                pass

    return None


def _get_text_config_dict(model_config) -> dict:
    """
    Returns the most relevant text config as a plain dict.
    Handles multimodal wrappers that store the real text config under text_config.
    """
    cfg = _pick_tp_config(model_config)

    # Prefer nested text_config when present
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        cfg = text_cfg

    # Convert HF config object -> plain dict safely
    if hasattr(cfg, "to_dict"):
        try:
            return cfg.to_dict()
        except Exception:
            pass

    # Fallback: shallow attr scrape
    result = {}
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            result[k] = v
    return result


def _is_qwen35_text_config(text_cfg: dict) -> bool:
    model_type = str(text_cfg.get("model_type", "")).lower()

    architectures = text_cfg.get("architectures") or []
    architectures = [str(x).lower() for x in architectures]

    if model_type == "qwen3_5_text":
        return True

    if "qwen3.5" in model_type or "qwen3_5" in model_type:
        return True

    if any("qwen3.5" in a or "qwen3_5" in a for a in architectures):
        return True

    return False


def _extract_qwen35_runtime_shapes_from_config(model_config) -> list[tuple[str, int, int, str]]:
    """
    Runtime-oriented GEMM shapes for Qwen3.5 text path.

    Returns:
        [(shard_type, N, K, label), ...]

    shard_type:
        - 'column' => TP shards N
        - 'row'    => TP shards K

    This intentionally mirrors the hardcoded runtime-derived logic you used
    before, including the two Qwen3.5-specific extras observed in vLLM warnings.
    """
    text_cfg = _get_text_config_dict(model_config)

    hidden_size = text_cfg.get("hidden_size")
    intermediate_size = text_cfg.get("intermediate_size")
    num_heads = text_cfg.get("num_attention_heads")
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim")

    if not hidden_size:
        return []

    shapes: list[tuple[str, int, int, str]] = []

    # ---- merged QKV ----
    # Q output = hidden_size
    # K output = num_kv_heads * head_dim
    # V output = num_kv_heads * head_dim
    if num_heads and num_kv_heads and head_dim:
        qkv_out = hidden_size + 2 * (num_kv_heads * head_dim)
        shapes.append(("column", int(qkv_out), int(hidden_size), "text_qkv_merged"))

    # ---- MLP ----
    if intermediate_size:
        merged_mlp_width = int(intermediate_size)
        mlp_down_k = merged_mlp_width // 2

        shapes.append(("column", merged_mlp_width, int(hidden_size), "text_mlp_expand"))
        shapes.append(("row", int(hidden_size), mlp_down_k, "text_mlp_down"))

    # ---- Qwen3.5-specific runtime extras ----
    # These are the special shapes you observed in actual vLLM runtime warnings.
    if _is_qwen35_text_config(text_cfg):
        shapes.append(("column", 8192, int(hidden_size), "text_qwen35_linear_attn_expand"))
        shapes.append(("row", int(hidden_size), 3072, "text_qwen35_linear_attn_down"))

    return shapes


def extract_runtime_gemm_shapes(model_path, tp_max: int = 1) -> list[tuple[int, int, str]]:
    model_path = str(model_path)

    # Load config first so we can detect model family
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # ------------------------------------------------------------
    # Qwen3.5 runtime-only path (short-circuit)
    # ------------------------------------------------------------
    qwen35_shapes = _extract_qwen35_runtime_shapes_from_config(config)

    if qwen35_shapes:
        found: "OrderedDict[tuple[int, int], set[str]]" = OrderedDict()
        print("Qwen3.5 detected")

        for shard_type, base_n, base_k, label in qwen35_shapes:
            for tp in _tp_levels(tp_max):
                n, k = base_n, base_k

                if shard_type == "column":
                    n = ceil_div(n, tp)
                elif shard_type == "row":
                    k = ceil_div(k, tp)
                else:
                    continue

                if min(n, k) < 1024:
                    continue

                found.setdefault((n, k), set()).add(f"{label}|{shard_type}|tp{tp}")

        return [
            (n, k, "|".join(sorted(labels)))
            for (n, k), labels in sorted(found.items())
        ]

    # ------------------------------------------------------------
    # Generic fallback path for other models
    # ------------------------------------------------------------
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    tp_cfg = _pick_tp_config(model.config)
    tp_plan = dict(getattr(tp_cfg, "base_model_tp_plan", {}) or {})

    found: "OrderedDict[tuple[int, int], set[str]]" = OrderedDict()

    for name, module in model.named_modules():
        shape = _linear_shape(module)
        if shape is None:
            continue

        base_n, base_k = shape

        if min(base_n, base_k) < 1024:
            continue

        tp_kind = _tp_kind_for_name(name, tp_plan)

        if tp_plan:
            if tp_kind is None:
                continue
            if tp_kind not in TP_COLWISE and tp_kind not in TP_ROWWISE:
                continue
        else:
            tp_kind = _fallback_tp_kind(name)
            if tp_kind is None:
                continue

        for tp in _tp_levels(tp_max):
            n, k = base_n, base_k

            if tp_kind in TP_COLWISE:
                if n % tp != 0:
                    continue
                n //= tp
            elif tp_kind in TP_ROWWISE:
                if k % tp != 0:
                    continue
                k //= tp

            found.setdefault((n, k), set()).add(f"{name}|{tp_kind}|tp{tp}")

    return [
        (n, k, "|".join(sorted(labels)))
        for (n, k), labels in sorted(found.items())
    ]