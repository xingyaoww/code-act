#!/bin/bash

YOUR_DOCKER_IMAGE_PATH=$1  # e.g., xingyaoww/codeact-executor for docker.io
if [ -z "$YOUR_DOCKER_IMAGE_PATH" ]; then
    YOUR_DOCKER_IMAGE_PATH=xingyaoww/codeact-executor
fi
echo "YOUR_DOCKER_IMAGE_PATH=$YOUR_DOCKER_IMAGE_PATH"
pushd scripts/chat/code_execution
docker build -f Dockerfile.executor -t $YOUR_DOCKER_IMAGE_PATH .
docker push $YOUR_DOCKER_IMAGE_PATH
popd
