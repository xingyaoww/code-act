#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent
echo_status

OUTPUT_DIR=$1 # "data/ckpts/Llama-2-7b-megatron-tp2-pp2/hf/mint_agent_iter_52"
MODEL_NAME="code-act-agent"

OUTPUT_DIR=$OUTPUT_DIR/eval/mt-bench
check_is_done $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

check_var "OPENAI_API_BASE"

# absolute path to the output directory
OUTPUT_DIR=$(realpath $OUTPUT_DIR)
mkdir -p $OUTPUT_DIR/model_answer
set -xe
pushd scripts/eval/mt-bench/FastChat/fastchat/llm_judge

OPENAI_API_KEY_OLD=$OPENAI_API_KEY
export OPENAI_API_KEY="DUMMY"
python3 gen_api_answer.py \
    --model $MODEL_NAME \
    --answer-file $OUTPUT_DIR/model_answer/mt-bench.jsonl

# NOTE: You need to pay
export OPENAI_API_KEY=$OPENAI_API_KEY_OLD
export OPENAI_API_BASE="https://api.openai.com/v1"
mkdir $OUTPUT_DIR/judgement
python gen_judgment.py \
    --answer-dir $OUTPUT_DIR/model_answer \
    --output-dir $OUTPUT_DIR/judgement \
    --judge-model gpt-4

python show_result.py \
    --input-file $OUTPUT_DIR/judgement/gpt-4_single.jsonl \
    --mode single > $OUTPUT_DIR/mt-bench-result.txt
popd
# Mark the evaluation as finished
touch $OUTPUT_DIR/DONE
