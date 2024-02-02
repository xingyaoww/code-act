#!/bin/bash

set -xe

TOKENIZER_PATH=data/models/raw/Llama-2-7b-megatron/tokenizer.model

# downloaded from huggingface following docs/MODEL_TRAINING.md
DATA_INPUT="data/datasets/codeact.jsonl data/datasets/general.jsonl"

# NOTE: You should use the following IF you are manually processing the dataset.
# DATA_INPUT="data/datasets/oct28_full6728.jsonl data/datasets/nov2_gpt4hard411.jsonl data/datasets/sharegpt_gpt4_all.jsonl data/datasets/sharegpt.n10000.jsonl data/datasets/openorca.n50000.jsonl data/datasets/capybara.jsonl"

OUTPUT_PREFX=data/megatron_format/nov9_mint+sharegpt4f_10k+openorca50k+capybara_pack4096/data
mkdir -p $(dirname $OUTPUT_PREFX)
python Megatron-LLM/tools/preprocess_instruct_data.py \
    --input $DATA_INPUT \
    --output_prefix $OUTPUT_PREFX \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file $TOKENIZER_PATH \
    --vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	--chunk_size 32 \
	--workers 16 \
    --no_new_tokens \
    --do_packing \
    --max_seq_length 4096
# Packed 78385 documents into 19379 documents.
