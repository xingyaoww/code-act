#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
pushd scripts/eval/m3tooleval

function run_eval {
    MODEL_NAME=$1
    echo "Running $MODEL_NAME"
    python3 main.py \
        --model $MODEL_NAME \
        --output_dir outputs/travel_itinerary_planning \
        --task_regex_filter "travel_itinerary_planning.*" \
        --n_turns_limit 10
    
    python3 main.py \
        --model $MODEL_NAME \
        --output_dir outputs/message_decoder \
        --task_regex_filter "message_decoder.*" \
        --n_turns_limit 10
    
    python3 main.py \
        --model $MODEL_NAME \
        --output_dir outputs/dna_sequencer \
        --task_regex_filter "dna_sequencer.*" \
        --n_turns_limit 10

    python3 main.py \
        --model $MODEL_NAME \
        --output_dir outputs/trade_calculator \
        --task_regex_filter "trade_calculator.*" \
        --n_turns_limit 10

    python3 main.py \
        --model $MODEL_NAME \
        --output_dir outputs/web_browsing \
        --task_regex_filter "web_browsing.*" \
        --n_turns_limit 10
}

# run_eval gpt-3.5-turbo-0301
# run_eval gpt-3.5-turbo-1106

# run_eval gpt-4-0613
# run_eval gpt-4-1106-preview

# run_eval claude-instant-1  # claude-instant-1.2
# run_eval claude-2 # claude-2.1

# run_eval text-davinci-003
# run_eval text-davinci-002

# NOTE: You may change the API endpoint to your own OpenAI-Complete API endpoint
# using vLLM: vllm.ai
export OPENAI_API_BASE=http://YOUR_API:8888/v1
run_eval YOUR_MODEL

popd
