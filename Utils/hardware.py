import os
import sys
import subprocess

def validate_and_restrict_gpus(gpus_arg: str) -> tuple[int, bool, str]:
    """
    Restricts GPU visibility if requested, and strictly validates that all 
    visible GPUs are identical. Halts the program if a heterogeneous environment is detected.
    Returns: (number_of_detected_gpus, is_amd, vendor_name)
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
        return 0, False, "unknown"

    visible_devices = []
    vendor = "unknown"
    is_amd = False

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            visible_devices.append(torch.cuda.get_device_name(i))
        # Check if it's AMD using torch.version.hip
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            is_amd = True
            vendor = "amd"
        else:
            vendor = "nvidia"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            visible_devices.append(torch.xpu.get_device_name(i))
        vendor = "intel"

    if not visible_devices:
        print("Warning: No GPUs detected by PyTorch. Proceeding without validation.")
        return 0, False, "unknown"

    unique_devices = set(visible_devices)

    if len(unique_devices) > 1:
        print("\n❌ Error: Heterogeneous GPU environment detected!")
        print("This tuning process requires all participating GPUs to be identical to guarantee accurate kernel selection.")
        print(f"Detected the following mixed GPUs: {', '.join(unique_devices)}")
        print("\nPlease restrict the environment to identical GPUs using the --gpus argument.")
        sys.exit(1)
    
    print(f"✅ GPU Validation Passed: {len(visible_devices)} identical GPU(s) found -> {list(unique_devices)[0]}")
    return len(visible_devices), is_amd, vendor

def get_amd_gfx_version(gfx_arg: str) -> str:
    """
    Detects the AMD gfx architecture. If multiple exist or there is ambiguity, 
    --gfx must be provided.
    """
    targets = set()
    try:
        output = subprocess.check_output(["rocminfo"], text=True)
        for line in output.split("\n"):
            if "Name:" in line and "gfx" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("gfx"):
                    targets.add(parts[1])
    except Exception:
        pass
        
    if not targets:
        if gfx_arg:
            return gfx_arg
        print("\n❌ Error: No gfx architectures detected via rocminfo, or could not run rocminfo.")
        print("Please provide the --gfx argument explicitly, e.g. --gfx gfx90a")
        sys.exit(1)
        
    if gfx_arg:
        if gfx_arg not in targets:
            print(f"\n❌ Error: The provided --gfx {gfx_arg} was not detected.")
            print(f"Detected targets: {', '.join(targets)}")
            sys.exit(1)
        return gfx_arg
        
    if len(targets) > 1:
        print(f"\n❌ Error: Multiple gfx architectures detected: {', '.join(targets)}")
        print("Please provide the precise --gfx argument, e.g. --gfx gfx90a")
        sys.exit(1)
        
    return list(targets)[0]

def validate_backend(backend: str, is_amd: bool):
    if backend == "aiter":
        print("\n⚠️ Warning: Pure AITER tuning is requested.")
        print("Pure AITER tuning is currently not allowed in this phase.")
        print("Only explicit Triton or explicit AITER-with-Triton-fallback paths are allowed.")
        print("Exiting cleanly.")
        sys.exit(0)

    if backend == "aiter_triton" and not is_amd:
        print("\n❌ Error: 'aiter_triton' is AMD-only.")
        sys.exit(1)

    if backend == "aiter_triton":
        try:
            import aiter
            import aiter.ops.triton
            print(f"[Backend Preflight] AITER package loadable: {aiter.__file__}")
            print(f"[Backend Preflight] AITER Triton module loadable: {aiter.ops.triton.__file__}")
        except Exception as e:
            print(f"\n❌ Error: AITER Triton preflight failed. Reason: {e}")
            sys.exit(1)

    try:
        import triton
        assert hasattr(triton, "jit"), "Triton missing jit"
        print(f"[Backend Preflight] Triton loadable: {triton.__file__}")
    except Exception as e:
        print(f"\n❌ Error: Triton preflight failed. Reason: {e}")
        sys.exit(1)

    print(f"\n=== Requested Backend: {backend} ===")
    print("Runtime backend proof will be enforced inside each isolated subprocess.")
