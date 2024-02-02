#!/bin/bash

RAW_MODEL_WEIGHT_DIR=data/models/raw_hf/Mistral-7B-v0.1/
OUTPUT_DIR=data/models/raw/Mistral-7b-megatron

python Megatron-LLM/weights_conversion/hf_to_megatron.py mistral \
    --size=7 \
	--out=$OUTPUT_DIR \
    --model-path=$RAW_MODEL_WEIGHT_DIR \
