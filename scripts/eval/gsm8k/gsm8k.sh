#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1 # "data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52"
MODEL_NAME="code-act-agent"

OUTPUT_DIR=$OUTPUT_DIR/eval/gsm8k
check_is_done $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

check_var "OPENAI_API_BASE"

set -xe
python3 scripts/eval/gsm8k/evaluate_gsm8k.py \
    --model $MODEL_NAME \
    --sample-input-file data/eval/gsm8k \
    --sample-output-file $OUTPUT_DIR/gsm8k_res.jsonl \

# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE
