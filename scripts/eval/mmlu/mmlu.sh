#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1 # "data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52"
MODEL_NAME="code-act-agent"

OUTPUT_DIR=$OUTPUT_DIR/eval/mmlu
check_is_done $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

check_var "OPENAI_API_BASE"

set -xe
python3 -u scripts/eval/mmlu/evaluate_mmlu.py \
    --model $MODEL_NAME \
    --data_dir data/eval/mmlu \
    --save_dir $OUTPUT_DIR

# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE
