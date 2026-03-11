import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python discover_vllm_kernels.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    print("Starting kernel discovery for:", model_path)

    # Import vLLM normally
    import vllm
    from vllm import LLM

    import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    captured = set()

    # We discovered this earlier from your scan
    original_fn = fp8_utils.get_w8a8_block_fp8_configs

    def hooked(N, K, device_name, dtype, block_shape, *args, **kwargs):
        filename = (
            f"N={N},K={K},device_name={device_name},"
            f"dtype={dtype},block_shape={block_shape}.json"
        )

        captured.add((N, K, device_name, dtype, tuple(block_shape), filename))

        return original_fn(N, K, device_name, dtype, block_shape, *args, **kwargs)

    # Patch vLLM
    fp8_utils.get_w8a8_block_fp8_configs = hooked

    print("Launching vLLM...")

    llm = LLM(
        model=model_path,
        max_model_len=4096,
        disable_log_stats=True,
    )

    print("Triggering minimal inference...")

    llm.generate("Hello", max_tokens=1)

    print("\n===================================")
    print("DISCOVERED KERNEL CONFIG REQUESTS")
    print("===================================\n")

    shapes = sorted(captured)

    for N, K, device, dtype, block_shape, filename in shapes:
        print("Shape:", N, K)
        print("JSON :", filename)
        print()

    print("Total unique kernels:", len(shapes))


if __name__ == "__main__":
    main()