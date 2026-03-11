import argparse
import os
import sys
import datetime
import json
from pathlib import Path
import time
import traceback

# Imps from Utils
from Utils.vllm_config_utils import (
    DTYPE_CONFIGS,
    find_vllm_base_path,
    load_json,
    try_get_device_name
)
from Utils.hardware import (
    validate_and_restrict_gpus,
    get_amd_gfx_version,
    validate_backend
)
from Utils.filesystem import (
    get_base_output_dir,
    setup_output_directories
)
from Utils.baselines import (
    get_baseline_file_path,
    BaselineCache
)

# Imps from Tuning
from tuning.inventory import generate_inventory, resolve_inventory_paths
from tuning.candidates import DenseCandidateBuilder, MoECandidateBuilder, mutate_candidate
from tuning.validator import validate_correctness, validate_minimal_runtime
from tuning.benchmarker import run_workload_profiles
from tuning.selector import score_and_select_winners, fallback_select_best_candidate, CalibrationSelectionError
from serialization.writer import write_batch_keyed_json, write_manifest

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MagicKernelConfigs Calibration System - Phase 2"
    )
    parser.add_argument("model_path", type=str, help="Path to model directory containing config.json")
    parser.add_argument("--vllm-path", type=str, default=None, help="Optional explicit vLLM package root")
    parser.add_argument("--dtype", type=str, choices=["fp8", "fp4", "int8", "int4"], default="fp8", help="Target precision data type")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of identical GPUs to use")
    parser.add_argument("--tp-max", type=int, default=512, help="Max tensor parallel size to inspect")
    parser.add_argument("--block-n", type=int, default=128, help="Block N size")
    parser.add_argument("--block-k", type=int, default=128, help="Block K size")
    parser.add_argument("--show-labels", action="store_true", help="Show source labels for each shape")
    parser.add_argument("--backend", type=str, choices=["triton", "aiter_triton"], required=True, help="Target backend")
    parser.add_argument("--gfx", type=str, default=None, help="gfx architecture (AMD only)")
    parser.add_argument("--baseline_path", type=str, default=None, help="path to baseline saves")
    args = parser.parse_args()

    # 1. Hardware Detection & Validation
    gpu_count, is_amd, vendor = validate_and_restrict_gpus(args.gpus)
    validate_backend(args.backend, is_amd)
    
    gfx_version = None
    if is_amd:
        gfx_version = get_amd_gfx_version(args.gfx)

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
    
    warnings_log = []
    if gpu_count > 0 and args.tp_max > gpu_count:
        warn_msg = f"Warning: tp-max ({args.tp_max}) > detected GPUs ({gpu_count}). Memory-pressure realism is incomplete."
        print(f"\n⚠️ {warn_msg}")
        warnings_log.append(warn_msg)

    # try_get_device_name currently returns the vLLM-style runtime device identifier,
    # which may already be normalized with underscores.
    try:
        device_name = try_get_device_name(vllm_base, dt_cfg["utils_file"])
    except TypeError:
        # the original util didn't handle overrides properly, fallback if needed
        device_name = "Unknown_Device"

    print(f"Device name: {device_name}")

    # Decide FP8 subtypes (currently implicitly generic e4m3 but architected to distinguish)
    is_fp8 = args.dtype == "fp8"
    fp8_subtype = "fp8_e4m3" if is_fp8 else None

    # Baseline Persistence Setup
    baseline_path_file = get_baseline_file_path(args.baseline_path, device_name, gfx_version, args.backend, args.dtype, fp8_subtype)
    baseline_cache = BaselineCache(baseline_path_file)
    if not baseline_cache.is_compatible(
        device_name, gfx_version, args.backend, args.dtype, fp8_subtype,
        args.block_n, args.block_k
    ):
        print("Warning: Existing baseline cache is from an incompatible version or hardware config. Resetting.")
        baseline_cache.clear()
        
    baseline_cache.init_metadata(
        gpu=device_name,
        gfx=gfx_version,
        backend=args.backend,
        dtype_family=args.dtype,
        dtype_subtype=fp8_subtype,
        block_n=args.block_n,
        block_k=args.block_k
    )
    print(f"\nBaseline cache location:\n{baseline_path_file}")

    # 2. Output directory scaffolding
    base_dir = get_base_output_dir(model_path, vendor, device_name, gfx_version, args.backend, args.dtype, args.tp_max)
    dense_dir, moe_dir = setup_output_directories(base_dir)

    # 3. Phase A - Target Preparation
    print(f"\n=== Generating Config Inventory for {args.dtype.upper()} ===")
    full_inventory = generate_inventory(
        model_path=model_path, 
        tp_target=args.tp_max, 
        device_name=device_name, 
        block_n=args.block_n, 
        block_k=args.block_k, 
        dtype_label=dt_cfg["file_label"]
    )
    
    active_inventory = resolve_inventory_paths(full_inventory, dense_dir, moe_dir)
    print(f"Total Targets: {len(full_inventory)} | Skipping: {len(full_inventory) - len(active_inventory)} | Tuning: {len(active_inventory)}")
    
    if not active_inventory:
         print("\nAll required configurations exist. Exiting cleanly.")
         sys.exit(0)         

    print(f"\n--- Launching Tuning Phase ---")
    print(f" Backend Configuration:   {args.backend}")
    print(f" DType Family Confirmed:  {args.dtype} (Subtype: {fp8_subtype})")
    print(f" Vendor / Platform:       {vendor.upper()} / {device_name}")
    print(f"------------------------------")

    dense_count = sum(1 for item in active_inventory if not item.is_moe)
    moe_count = sum(1 for item in active_inventory if item.is_moe)
    
    for i, item in enumerate(active_inventory, start=1):
        print(f"\n[{i}/{len(active_inventory)}] Tuning -> {item.filename}")
        if item.is_moe:
            builder = MoECandidateBuilder(
                backend=args.backend,
                vendor=vendor,
                target_n=item.n,
                target_k=item.k
            )
        else:
            builder = DenseCandidateBuilder(
                is_amd=is_amd,
                is_fp8=is_fp8,
                backend=args.backend,
                vendor=vendor,
                target_n=item.n,
                target_k=item.k
            )
            
        candidates = builder.build()
        print(f"  Generated {len(candidates)} pruned baseline candidates.")
        baseline_cache.setup_shape(item.n, item.k, item.is_moe, len(candidates))

        progress_log = item.output_path.with_name(f"progress_{item.output_path.stem}.jsonl")
        bench_dump_path = item.output_path.with_name(f"{item.output_path.stem}.bench.json")
        
        # Repair Detection
        needs_repair = False
        if not item.output_path.exists():
            needs_repair = True
        elif item.output_path.stat().st_size == 0:
            needs_repair = True
        else:
            try:
                with open(item.output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data:
                        needs_repair = True
            except json.JSONDecodeError:
                needs_repair = True
                
        if needs_repair and item.output_path.exists():
            corrupt_path = item.output_path.with_name(f"{item.output_path.name}.corrupt")
            item.output_path.rename(corrupt_path)
            print(f"  ⚠️ Invalid output detected. Renamed to {corrupt_path.name} to trigger repair.")
            append_jsonl(progress_log, {"event": "repair_detected", "filename": item.output_path.name})
        
        if not needs_repair:
            # We trust inventory generation, but this makes it explicitly safe
            print(f"  ✅ Output {item.filename} looks valid. Skipping.")
            continue
            
        candidate_results = []
        is_resume = False
        
        # Pre-Validate Benchmark Schema & Load Cache
        if bench_dump_path.exists():
            try:
                with open(bench_dump_path, 'r', encoding='utf-8') as f:
                    bench_data = json.load(f)
                    if isinstance(bench_data, list) and all(isinstance(x, dict) and "candidate" in x and "profiles" in x for x in bench_data):
                        print(f"  📦 Resuming from valid benchmark data: {bench_dump_path.name}")
                        is_resume = True
                        for entry in bench_data:
                            k = candidate_key(entry["candidate"])
                            baseline_cache.add_result(item.n, item.k, item.is_moe, k, entry)
                    else:
                        print(f"  ⚠️ Benchmark data malformed. Ignoring {bench_dump_path.name}.")
            except Exception as e:
                print(f"  ⚠️ Could not read benchmark data: {e}")

        # Phase D/E - Orchestrate execution across GPU workloads
        seen_candidates = set()

        total_baselines = len(candidates)
        baseline_done = 0
        survivors = 0

        print(f"  🚀 Launching Profiler ({args.backend})...", flush=True)

        for c_idx, candidate in enumerate(candidates, start=1):
            baseline_done += 1
            base_label = f"baseline {baseline_done}/{total_baselines}"
            base_key = candidate_key(candidate)

            if base_key in seen_candidates:
                append_jsonl(progress_log, {
                    "event": "baseline_skipped_duplicate",
                    "label": base_label,
                    "candidate": candidate,
                })
                print(f"    ↷ {base_label} duplicate baseline skipped", flush=True)
                continue

            seen_candidates.add(base_key)

            print(
                f"  ⚙️  {base_label} | current survivors={survivors} | "
                f"candidate={short_candidate(candidate)}",
                flush=True
            )

            if baseline_cache.has_candidate(item.n, item.k, item.is_moe, base_key):
                print(f"    📦 Loading {base_label} from baseline cache...", flush=True)
                shape_key = baseline_cache.get_shape_key(item.n, item.k, item.is_moe)
                base_result = baseline_cache.data["shapes"][shape_key]["results"][base_key]
            else:
                base_result = evaluate_candidate_with_logs(
                    candidate=candidate,
                    label=base_label,
                    item=item,
                    args=args,
                    gpu_count=gpu_count,
                    progress_log=progress_log,
                )
                if base_result is not None:
                    baseline_cache.add_result(item.n, item.k, item.is_moe, base_key, base_result)

            if base_result is not None:
                candidate_results.append(base_result)
                survivors += 1

                # Mutation exploration only after a valid baseline
                mutations = mutate_candidate(
                    candidate,
                    item.is_moe,
                    args.backend,
                    vendor,
                    item.n,
                    item.k
                )
                append_jsonl(progress_log, {
                    "event": "mutations_generated",
                    "label": base_label,
                    "count": len(mutations),
                })
                print(f"      generated {len(mutations)} mutations", flush=True)

                for m_idx, m_cand in enumerate(mutations, start=1):
                    mut_label = f"{base_label} -> mut {m_idx}/{len(mutations)}"
                    m_key = candidate_key(m_cand)

                    if m_key in seen_candidates:
                        append_jsonl(progress_log, {
                            "event": "mutation_skipped_duplicate",
                            "label": mut_label,
                            "candidate": m_cand,
                        })
                        print(f"      ↷ duplicate mutation skipped ({m_idx}/{len(mutations)})", flush=True)
                        continue

                    seen_candidates.add(m_key)

                    if baseline_cache.has_candidate(item.n, item.k, item.is_moe, m_key):
                        print(f"      📦 Loading mutation ({m_idx}/{len(mutations)}) from baseline cache...", flush=True)
                        shape_key = baseline_cache.get_shape_key(item.n, item.k, item.is_moe)
                        mut_result = baseline_cache.data["shapes"][shape_key]["results"][m_key]
                    else:
                        mut_result = evaluate_candidate_with_logs(
                            candidate=m_cand,
                            label=mut_label,
                            item=item,
                            args=args,
                            gpu_count=gpu_count,
                            progress_log=progress_log,
                        )
                        if mut_result is not None:
                            baseline_cache.add_result(item.n, item.k, item.is_moe, m_key, mut_result)

                    if mut_result is not None:
                        candidate_results.append(mut_result)
                        survivors += 1

            append_jsonl(progress_log, {
                "event": "baseline_complete",
                "label": base_label,
                "survivors_so_far": survivors,
                "candidate_results_so_far": len(candidate_results),
            })

        print(f"\n  ✅ Benchmarking complete. {survivors} viable parameter variations survived.", flush=True)
        print(f"  📝 Progress log: {progress_log}", flush=True)
                     
        if not candidate_results:
             print(f"  ❌ No candidates survived validation bounding for {item.filename}.")
             continue
             
        # Phase F - Selection
        winners_by_bucket = score_and_select_winners(candidate_results)
        if not winners_by_bucket:
             print("  ⚠️ Standard selector yielded empty buckets. Engaging fallback_select_best_candidate()...")
             winners_by_bucket = fallback_select_best_candidate(candidate_results)
             
        if not winners_by_bucket:
             raise CalibrationSelectionError(f"Calibration for {item.filename} failed to produce any valid output buckets.")
        
        # Save raw benchmarks logs before paring down winners
        try:
            with open(bench_dump_path, 'w', encoding='utf-8') as f:
                json.dump(candidate_results, f, indent=4)
            if is_resume:
                print(f"  📝 Updated complete benchmark tracking to: {bench_dump_path.name}")
            else:
                print(f"  📝 Saved complete benchmark tracking to: {bench_dump_path.name}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not save benchmark dump: {e}")
            
        # Phase G - Persistence
        write_batch_keyed_json(winners_by_bucket, item.output_path, item.is_moe, args.backend)
        
        if is_resume:
            append_jsonl(progress_log, {
                "event": "repair_reconstructed_from_benchmarks",
                "winners": len(winners_by_bucket)
            })
        
        # Enforce validation of the JSON format against exact loader rules
        if not validate_minimal_runtime(item.output_path, args.tp_max):
             print(f"  ❌ Fatal Error: Post-write validation failed for {item.filename}.")
             sys.exit(1)
             
        print(f"  ✅ Saved dynamically scaled configs mapping to buckets: {list(winners_by_bucket.keys())}")
             
    # Write global manifest securely tracking context
    manifest_data = {
        "model_path": str(model_path),
        "dtype_family": args.dtype,
        "dtype_subtype": "fp8_e4m3" if args.dtype == "fp8" else None,
        "backend": args.backend,
        "vendor": vendor,
        "normalized_gpu_name": device_name.replace(" ", "_"),
        "exact_runtime_device_name": device_name,
        "gfx_target": gfx_version,
        "tensor_parallel_target": args.tp_max,
        "actual_detected_gpu_count": gpu_count,
        "selected_gpu_list": args.gpus,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "dense_target_count": dense_count,
        "moe_target_count": moe_count,
        "tuning_timestamp": datetime.datetime.now().isoformat(),
        "tuning_status_summary": "Phase 2 Automated Run Completed",
        "warnings": warnings_log
    }
    write_manifest(base_dir / "manifest.json", manifest_data)
    print(f"\n=== Calibration Loop Complete. Manifest saved to {base_dir / 'manifest.json'} ===")

def candidate_key(candidate) -> str:
    """Stable-ish key for dedupe/logging."""
    try:
        return json.dumps(candidate, sort_keys=True, default=str)
    except Exception:
        return repr(candidate)

def append_jsonl(path: Path, payload: dict) -> None:
    row = {
        "ts": datetime.datetime.now().isoformat(),
        **payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()

def short_candidate(candidate, max_len: int = 220) -> str:
    s = candidate_key(candidate)
    return s if len(s) <= max_len else s[:max_len] + "..."

def evaluate_candidate_with_logs(
    *,
    candidate,
    label: str,
    item,
    args,
    gpu_count: int,
    progress_log: Path,
):
    started = time.monotonic()
    append_jsonl(progress_log, {
        "event": "candidate_start",
        "label": label,
        "filename": item.filename,
        "candidate": candidate,
    })
    print(f"    ▶ {label} START", flush=True)

    try:
        t0 = time.monotonic()
        ok = validate_correctness(candidate, item.n, item.k, 16, args.dtype, item.is_moe, args.backend)
        t1 = time.monotonic()

        append_jsonl(progress_log, {
            "event": "correctness_done",
            "label": label,
            "ok": ok,
            "seconds": round(t1 - t0, 4),
        })
        print(f"      correctness={ok} in {t1 - t0:.2f}s", flush=True)

        if not ok:
            append_jsonl(progress_log, {
                "event": "candidate_rejected",
                "label": label,
                "reason": "correctness_failed",
            })
            return None

        t2 = time.monotonic()
        profiles = run_workload_profiles(candidate, item.n, item.k, item.is_moe, gpu_count, args.backend, args.dtype)
        t3 = time.monotonic()

        append_jsonl(progress_log, {
            "event": "profiles_done",
            "label": label,
            "seconds": round(t3 - t2, 4),
            "profile_keys": list(profiles.keys()) if isinstance(profiles, dict) else None,
        })
        print(f"      profiles done in {t3 - t2:.2f}s", flush=True)

        total = time.monotonic() - started
        append_jsonl(progress_log, {
            "event": "candidate_success",
            "label": label,
            "total_seconds": round(total, 4),
        })
        print(f"    ✅ {label} SUCCESS in {total:.2f}s", flush=True)

        return {
            "candidate": candidate,
            "profiles": profiles,
        }

    except Exception as e:
        total = time.monotonic() - started
        tb = traceback.format_exc()
        append_jsonl(progress_log, {
            "event": "candidate_error",
            "label": label,
            "error": str(e),
            "traceback": tb,
            "total_seconds": round(total, 4),
        })
        print(f"    ❌ {label} ERROR after {total:.2f}s -> {e}", flush=True)
        return None

if __name__ == "__main__":
    main()