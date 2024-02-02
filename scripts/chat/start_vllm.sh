#!/bin/bash
# Require install docker and nvidia-docker2
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.8.0/install-guide.html

MODEL_PATH=$1
MODEL_DIR=$(dirname $MODEL_PATH)
MODEL_NAME=$(basename $MODEL_PATH) # CodeActAgent-Mistral-7b-v0.1
PORT=$2 # 8080
if [ -z "$PORT" ]; then
    PORT=8080
fi
echo "PORT=$PORT"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_DIR=$MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

docker run \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --runtime nvidia --gpus all \
    -v $MODEL_DIR:/data \
    -p $PORT:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --host 0.0.0.0 \
    --model /data/$MODEL_NAME \
    --served-model-name $MODEL_NAME \
