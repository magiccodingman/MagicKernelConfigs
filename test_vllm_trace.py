import logging
import argparse
from vllm import LLM
import os

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

try:
    llm = LLM(
        model="/home/slurp/Source/MagicCodingMan/MagicKernelConfigs/dummy_testing_model", 
        load_format="dummy", 
        tensor_parallel_size=1, 
        enforce_eager=True,
        dtype="float16" # Keep simple for testing
    )
except Exception as e:
    import traceback
    traceback.print_exc()
