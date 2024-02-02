
pushd scripts/eval/api-bank

mkdir -p results

function run_eval() {
    model_name=$1
    echo "Evaluating $model_name"
    for action_mode in "text_as_action" "json_as_action" "code_as_action"; do
        echo "===================="
        echo "action_mode: $action_mode"
        python3 evaluator.py \
            --data_dir lv1-lv2-samples/level-1-given-desc \
            --action_mode $action_mode \
            --model_name $model_name \
            --output_dir results \
            --api_test_enabled
    done
}

# run_eval gpt-3.5-turbo-0613
# run_eval gpt-3.5-turbo-1106

# run_eval gpt-4-0613
# run_eval gpt-4-1106-preview

# run_eval text-davinci-003
# run_eval text-davinci-002

# run_eval claude-instant-1  # claude-instant-1.2
# run_eval claude-2 # claude-2.1

# run_eval gemini-pro

# NOTE: You may change the API endpoint to your own OpenAI-Complete API endpoint
# using vLLM: vllm.ai
export OPENAI_API_BASE=http://YOUR_API:8888/v1
run_eval YOUR_MODEL

popd
