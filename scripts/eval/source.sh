#!/bin/bash

eval "$(conda shell.bash hook)"

function check_conda_env_and_activate() {
    name=$1
    env_exists=$(conda info --envs | grep "^$name\s")
    if [ -z "$env_exists" ]; then
        echo "Environment $name does not exist. Exiting."
        exit 1
    fi
    conda activate $name
}

function check_var() {
    name=$1
    if [ -z "${!name}" ]; then
        echo "Variable $name is not set. Exiting."
        exit 1
    fi
}

function check_model_is_served() {
    PORT=$1
    # Retrieve the list of models and check if it's empty
    response=$(curl -s http://localhost:$PORT/v1/models)
    models=$(echo "$response" | jq '.data[]')

    # If the list of models is empty, exit
    if [[ -z $models ]]; then
        echo "No models are being served on port $PORT."
        return 1
    fi

    # If the list of models is not empty, success
    return 0
}

function wait_for_model() {
    PORT=$1
    MAX_RETRIES=300
    RETRY_COUNT=0

    # Poll the HTTP endpoint until the worker registers or MAX_RETRIES is reached
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "Waiting for VLLM served model..."

        # if the model is served, break from the loop
        check_model_is_served $PORT && break

        # Increment the retry count
        RETRY_COUNT=$((RETRY_COUNT + 1))

        # Sleep for a short while before polling again
        sleep 5
    done

    # Check if we've exceeded max retries
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Max retries reached. Model may not be served."
        exit 1
    fi
}


function check_is_done() {
    OUTPUT_DIR=$1
    if [ -f "$OUTPUT_DIR/DONE" ]; then
        echo "Evaluation already finished for $OUTPUT_DIR. Skipping."
        exit 0
    fi
}


function echo_status() {
    # echo timestamp and hostname before the message
    echo "=============================================="
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S") Host:$(hostname)"
}

function kill_serving () {
    echo "Kill all processes owned by the current user $USER with keyword 'fastchat' and 'vllm'..."
    pkill -u $USER -f fastchat.serve
    pkill -u $USER -f vllm.entrypoints.openai.api_server
}

# Function to check if a port is occupied
function is_port_in_use() {
    netstat -tuln | grep -q ":$1\s"
    return $?
}

function remove_from_array {
    local -n arr=$1
    local value=$2

    # Find the index of the element to be removed
    local index=0
    for i in "${arr[@]}"; do
        if [ "$i" = "$value" ]; then
            break
        fi
        index=$((index + 1))
    done

    # Remove the element at the found index
    unset 'arr[$index]'

    # Reindex the array
    arr=("${arr[@]}")
}
