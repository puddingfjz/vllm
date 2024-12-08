"""
    This file contains the script to prepare our test dataset.
"""

# prepare the LLM-Blender dataset: mix-instruct

import json

# Path to your JSONL file
file_path = "/ssddata/jingzhi/vLLM/vllm/benchmarks/train_data_prepared.jsonl"

# Read the JSONL file
with open(file_path) as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line)
        a = data
        break

