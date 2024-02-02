#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1 # "data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52"
MODEL_NAME="code-act-agent"

OUTPUT_DIR=$OUTPUT_DIR/eval/human_eval
check_is_done $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

check_var "OPENAI_API_BASE"

set -xe
python3 scripts/eval/human_eval/evaluate_human_eval.py \
    --model $MODEL_NAME \
    --save_dir $OUTPUT_DIR

evaluate_functional_correctness $OUTPUT_DIR/samples.jsonl > $OUTPUT_DIR/result.txt

# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE
