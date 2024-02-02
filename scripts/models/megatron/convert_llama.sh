#!/bin/bash

RAW_MODEL_WEIGHT_DIR=data/models/raw_hf/Llama-2-7b-hf/
OUTPUT_DIR=data/models/raw/Llama-2-7b-megatron

python Megatron-LLM/weights_conversion/hf_to_megatron.py llama2 \
    --size=7 \
	--out=$OUTPUT_DIR \
    --model-path=$RAW_MODEL_WEIGHT_DIR \
