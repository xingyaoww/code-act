#!/bin/bash
# This script can be used to run commands inside the docker container.

mkdir -p data/outputs
mkdir -p data/processed

export HOST_USER_ID=$(id -u)
DOCKER_IMG="xingyaoww/mint-dev:v1.0"

# Construct instance name using the current username and the current time.
# This is useful for running multiple instances of the docker container.
DOCKER_INSTANCE_NAME="mint_${USER}_$(date +%Y%m%d_%H%M%S)"

docker run \
    -e HOST_USER_ID \
    -e OPENAI_API_KEY \
    -e BARD_API_KEY \
    -e ANTHROPIC_API_KEY \
    -it \
    -v `pwd`:/mint-bench:ro \
    -v `pwd`/data/outputs:/mint-bench/data/outputs:rw \
    -v `pwd`/data/raw:/mint-bench/data/raw:rw \
    -v `pwd`/data/processed:/mint-bench/data/processed:rw \
    --network host \
    --rm \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --gpus all \
    --name $DOCKER_INSTANCE_NAME \
    --ulimit nofile=4096:4096 \
    $DOCKER_IMG \
    bash -c "cd /mint-bench && useradd --shell /bin/bash -u $HOST_USER_ID -o -c "" -m mint && su mint"
