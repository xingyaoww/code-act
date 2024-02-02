#!/bin/bash
source scripts/eval/source.sh
check_conda_env_and_activate code-act-agent

set -m  # Enable job control

MODEL_DIR=$1
PORT=$2  # default to 8888
if [ -z "$PORT" ]; then
    PORT=8888
fi

# parse CUDA_VISIBLE_DEVICES to get number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=0
else
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    N_GPUS=${#GPU_IDS[@]}
fi

echo "Serve model from $MODEL_DIR with $N_GPUS GPUs on port $PORT"

while true; do
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --served-model-name code-act-agent \
        --tensor-parallel-size $N_GPUS \
        --host 0.0.0.0 --port $PORT \
        --swap-space 1
    echo "Server crashed, restarting in 5 seconds..."
    sleep 5
done
