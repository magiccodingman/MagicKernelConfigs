import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Imported from the new Utils directory
from Utils.vllm_config_utils import (
    DTYPE_CONFIGS,
    apply_tp_sharding,
    build_tp_levels,
    extract_qwen35_runtime_text_shapes,
    find_vllm_base_path,
    get_dtype_paths,
    load_json,
    make_filename,
    normalize_text_config,
    should_keep_shape,
    try_get_device_name,
)


def validate_and_restrict_gpus(gpus_arg: str) -> None:
    """
    Restricts GPU visibility if requested, and strictly validates that all 
    visible GPUs are identical. Halts the program if a heterogeneous environment is detected.
    """
    if gpus_arg:
        # Set environment variables to restrict visibility before Torch/vLLM initialize
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_arg
        os.environ["ROCR_VISIBLE_DEVICES"] = gpus_arg
        os.environ["HIP_VISIBLE_DEVICES"] = gpus_arg
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpus_arg}"

    try:
        import torch
    except ImportError:
        print("Warning: PyTorch is not installed. Skipping GPU safety validation.")
        return

    visible_devices = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            visible_devices.append(torch.cuda.get_device_name(i))
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            visible_devices.append(torch.xpu.get_device_name(i))

    if not visible_devices:
        print("Warning: No GPUs detected by PyTorch. Proceeding without validation.")
        return

    unique_devices = set(visible_devices)

    if len(unique_devices) > 1:
        print("\n❌ Error: Heterogeneous GPU environment detected!")
        print("This tuning process requires all participating GPUs to be identical to guarantee accurate kernel selection.")
        print(f"Detected the following mixed GPUs: {', '.join(unique_devices)}")
        print("\nPlease restrict the environment to identical GPUs using the --gpus argument.")
        print("Example: python main.py <model_path> --gpus 0,1")
        print("\nAlternatively, set the appropriate environment variable manually before running:")
        print("  - NVIDIA: export CUDA_VISIBLE_DEVICES=0,1")
        print("  - AMD:    export ROCR_VISIBLE_DEVICES=0,1 (or HIP_VISIBLE_DEVICES=0,1)")
        print("  - Intel:  export ONEAPI_DEVICE_SELECTOR=level_zero:0,1")
        sys.exit(1)
    
    print(f"✅ GPU Validation Passed: {len(visible_devices)} identical GPU(s) found -> {list(unique_devices)[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM Config Auditor for Qwen3.5 runtime-needed text kernels"
    )
    parser.add_argument("model_path", type=str, help="Path to model directory containing config.json")
    parser.add_argument("--vllm-path", type=str, default=None, help="Optional explicit vLLM package root")
    parser.add_argument("--dtype", type=str, choices=["fp8", "fp4", "int8", "int4"], default="fp8", help="Target precision data type (default: fp8)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of identical GPUs to use (e.g., '0,1'). Validates uniformity and isolates them.")
    parser.add_argument("--tp-max", type=int, default=512, help="Max tensor parallel size to inspect")
    parser.add_argument("--block-n", type=int, default=128, help="Block N size")
    parser.add_argument("--block-k", type=int, default=128, help="Block K size")
    parser.add_argument("--show-labels", action="store_true", help="Show source labels for each shape")
    args = parser.parse_args()

    # 1. Validate the GPU environment and apply restrictions immediately
    validate_and_restrict_gpus(args.gpus)

    model_path = Path(args.model_path)
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"Error: No config.json found in {model_path}")
        sys.exit(1)

    vllm_base = Path(args.vllm_path) if args.vllm_path else find_vllm_base_path()
    if not vllm_base.exists():
        print(f"Error: vLLM path does not exist: {vllm_base}")
        sys.exit(1)

    dt_cfg = DTYPE_CONFIGS[args.dtype]
    
    root_config = load_json(config_file)
    text_cfg = normalize_text_config(root_config)

    # 2. Automatically detect the formatted device name using vLLM's util (no override argument needed)
    # Note: Because we isolated the GPUs in validate_and_restrict_gpus, vLLM will safely auto-detect the identical cards
    device_name = try_get_device_name(vllm_base, dt_cfg["utils_file"])
    configs_dir, bench_script = get_dtype_paths(vllm_base, dt_cfg["utils_file"], dt_cfg["bench_script"])

    raw_shapes = extract_qwen35_runtime_text_shapes(root_config)
    tp_levels = build_tp_levels(args.tp_max)

    per_tp_results = {}
    all_missing_filenames = set()
    all_existing_filenames = set()
    all_missing_cmds = set()

    print(f"Auditing configs for {args.dtype.upper()} targeting {device_name}...")

    for tp in tp_levels:
        shaped = apply_tp_sharding(raw_shapes, tp)

        unique_by_shape: dict[tuple[int, int], set[str]] = defaultdict(set)
        for N, K, label in shaped:
            if should_keep_shape(N, K, args.block_n, args.block_k):
                unique_by_shape[(N, K)].add(label)

        existing = []
        missing = []

        for (N, K), labels in sorted(unique_by_shape.items()):
            fname = make_filename(N, K, device_name, args.block_n, args.block_k, dt_cfg["file_label"])
            entry = {
                "N": N,
                "K": K,
                "filename": fname,
                "labels": sorted(labels),
                "exists": (configs_dir / fname).exists(),
            }

            if entry["exists"]:
                existing.append(entry)
                all_existing_filenames.add(fname)
            else:
                missing.append(entry)
                all_missing_filenames.add(fname)
                all_missing_cmds.add(
                    f"python {bench_script} --N {N} --K {K} "
                    f"--device-name {device_name} --output-dir {configs_dir}"
                )

        per_tp_results[tp] = {
            "existing": existing,
            "missing": missing,
        }

    for tp in tp_levels:
        existing = per_tp_results[tp]["existing"]
        missing = per_tp_results[tp]["missing"]

        print(f"\n=== TENSOR PARALLEL {tp} ===")
        print(f"  ✅ Existing: {len(existing)}")
        print(f"  ❌ Missing:  {len(missing)}")

        for item in missing:
            if args.show_labels:
                print(f"    - {item['filename']}    [{', '.join(item['labels'])}]")
            else:
                print(f"    - {item['filename']}")

    print("\n=== GLOBAL UNIQUE SUMMARY ===")
    print(f"  ✅ Unique existing configs: {len(all_existing_filenames)}")
    print(f"  ❌ Unique missing configs:  {len(all_missing_filenames)}")

    tune_script = Path(f"tune_vllm_{args.dtype}.sh")
    with open(tune_script, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        
        # 3. Propagate the GPU isolation to the outputted bash script
        if args.gpus:
            f.write(f"# Restricting environment to identical GPUs selected during generation\n")
            f.write(f"export CUDA_VISIBLE_DEVICES={args.gpus}\n")
            f.write(f"export ROCR_VISIBLE_DEVICES={args.gpus}\n")
            f.write(f"export HIP_VISIBLE_DEVICES={args.gpus}\n")
            f.write(f"export ONEAPI_DEVICE_SELECTOR=level_zero:{args.gpus}\n\n")

        for cmd in sorted(all_missing_cmds):
            f.write(cmd + "\n")

    try:
        tune_script.chmod(0o755)
    except Exception:
        pass

    print(f"\n🚀 Saved {len(all_missing_cmds)} tuning commands to '{tune_script}'.")

    print("\n=== RAW DERIVED SHAPES (PRE-TP) ===")
    for shard_type, N, K, label in raw_shapes:
        print(f"  - {label}: shard={shard_type}, N={N}, K={K}")

    print("\n=== TEXT CONFIG SUMMARY ===")
    print(f"  hidden_size:         {text_cfg.get('hidden_size')}")
    print(f"  intermediate_size:   {text_cfg.get('intermediate_size')}")
    print(f"  num_attention_heads: {text_cfg.get('num_attention_heads')}")
    print(f"  num_key_value_heads: {text_cfg.get('num_key_value_heads')}")
    print(f"  head_dim:            {text_cfg.get('head_dim')}")
    print(f"  model_type:          {text_cfg.get('model_type')}")


if __name__ == "__main__":
    main()