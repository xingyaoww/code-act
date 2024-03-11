#!/bin/bash

YOUR_DOCKER_IMAGE_PATH=$1  # e.g., xingyaoww/codeact-execute-api for docker.io
if [ -z "$YOUR_DOCKER_IMAGE_PATH" ]; then
    YOUR_DOCKER_IMAGE_PATH=xingyaoww/codeact-execute-api
fi
echo "YOUR_DOCKER_IMAGE_PATH=$YOUR_DOCKER_IMAGE_PATH"
pushd scripts/chat/code_execution
docker build -f Dockerfile.api -t $YOUR_DOCKER_IMAGE_PATH .
docker push $YOUR_DOCKER_IMAGE_PATH
popd
