#!/bin/bash

echo "NOTE: You should configure ./chat-ui/.env.local first before running this script."

YOUR_DOCKER_IMAGE_PATH=$1  # e.g., xingyaoww/chat-ui for docker.io
if [ -z "$YOUR_DOCKER_IMAGE_PATH" ]; then
    YOUR_DOCKER_IMAGE_PATH=xingyaoww/chat-ui
fi
echo "YOUR_DOCKER_IMAGE_PATH=$YOUR_DOCKER_IMAGE_PATH"

pushd chat-ui
docker build -t $YOUR_DOCKER_IMAGE_PATH --secret id=DOTENV_LOCAL,src=.env.local .
docker push $YOUR_DOCKER_IMAGE_PATH
popd
