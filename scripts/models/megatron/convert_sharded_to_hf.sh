#!/bin/bash

# scripts/train/megatron/convert_sharded_to_hf.sh data/ckpts/Llama-2-7b-megatron-tp2-pp2 52
LOAD_DIR=$1 # data/ckpts/Llama-2-7b-megatron-tp2-pp2
LOAD_ITER=$2 # 300 to load the from iter_0000300 iteration
OUTPUT_DIR=$LOAD_DIR"/hf/model_iter_"$LOAD_ITER
echo "Converting sharded checkpoint $LOAD_DIR to huggingface $OUTPUT_DIR"

if [[ "$LOAD_DIR" == *Llama-2* ]]; then
    MODEL_TYPE=llama2
    MODEL_NAME=Llama-2-7b
elif [[ "$LOAD_DIR" == *Mistral-7b* ]]; then
    MODEL_TYPE=mistral
    MODEL_NAME=Mistral-7b
else
    echo "Unknown model name $MODEL_NAME"
    exit 1
fi

echo "MODEL_TYPE=$MODEL_TYPE"
echo "MODEL_NAME=$MODEL_NAME"

TOKENIZER_PATH=data/models/raw/${MODEL_NAME}-megatron/tokenizer.model
VOCAB_SIZE=32002  # 32000 + 2 for --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
echo "Use vocab file $TOKENIZER_PATH"

TMP_DIR=tmp/megatron-conversion
TMP_OUTPUT_DIR=$TMP_DIR/$LOAD_DIR
mkdir -p $TMP_OUTPUT_DIR

set -ex
# convert to unsharded checkpoint
python Megatron-LLM/tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir $LOAD_DIR \
	--load_iters $LOAD_ITER \
	--save_dir $TMP_OUTPUT_DIR \
	--model_type $MODEL_TYPE \
	--true_vocab_size $VOCAB_SIZE \
	--use_distributed_optimizer \
	--bf16

# convert to huggingface checkpoint
python Megatron-LLM/weights_conversion/megatron_to_hf.py \
	--model=$MODEL_TYPE \
	--input_dir=$TMP_OUTPUT_DIR \
	--output_dir=$OUTPUT_DIR \
	--vocab_file=$TOKENIZER_PATH \
	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	--no_new_tokens

# Add chat format
python3 scripts/chat_interface/add_chat_format.py $OUTPUT_DIR

rm -rf $TMP_DIR
