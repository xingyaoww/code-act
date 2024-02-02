#!/bin/bash
source scripts/train/megatron/source.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3

TP=4
PP=1
LR=1e-5
N_WARMUP_STEPS=50
SEQ_LENGTH=16384
MODEL_NAME=Mistral-7b

# TODO: CHANGE THIS FOR DIFFERENT DATASET
DATA_NAME=nov9_mint+sharegpt4f_10k+openorca50k+capybara_pack16k_mistral
N_EXAMPLES=4833

GLOBAL_BATCH_SIZE=32
N_EPOCHS=5
N_GPU_PER_NODE=4

train \
 --tp $TP \
 --pp $PP \
 --lr $LR \
 --n_warmup_steps $N_WARMUP_STEPS \
 --seq_length $SEQ_LENGTH \
 --model_name $MODEL_NAME \
 --data_name $DATA_NAME \
 --n_examples $N_EXAMPLES \
 --global_batch_size $GLOBAL_BATCH_SIZE \
 --n_epochs $N_EPOCHS \
 --n_gpu_per_node $N_GPU_PER_NODE \
 --recompute_num_layers 2 \
 --use_flash_attn \
 --packed_input
