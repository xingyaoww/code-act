#!/bin/bash

TP=4
PP=1
MODEL_NAME=$1 # Llama-2-7b
# convert model name to type
# Llama-2-* -> llama2
# Mistral-* -> mistral
if [[ $MODEL_NAME == *"Llama"* ]]; then
	MODEL_TYPE=llama2
elif [[ $MODEL_NAME == *"Mistral"* ]]; then
	MODEL_TYPE=mistral
else
	echo "Error: Unknown model name $MODEL_NAME"
	exit 1
fi
echo "MODEL_TYPE=$MODEL_TYPE"

MODEL_DIR=data/models/raw/${MODEL_NAME}-megatron
OUTPUT_DIR=data/models/sharded/${MODEL_NAME}-megatron-tp${TP}-pp${PP}

VOCAB_PATH=data/models/raw/${MODEL_NAME}-megatron/tokenizer.model
VOCAB_SIZE=32002  # 32000 + 2 for --vocab_extra_ids_list "<|im_start|>,<|im_end|>"

python Megatron-LLM/tools/checkpoint_util.py \
	--target_tensor_parallel_size $TP \
	--target_pipeline_parallel_size $PP \
	--load_dir $MODEL_DIR \
	--save_dir $OUTPUT_DIR \
	--model_type $MODEL_TYPE \
	--true_vocab_size $VOCAB_SIZE \
	--bf16
