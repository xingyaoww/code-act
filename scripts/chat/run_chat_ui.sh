#!/bin/bash

echo "NOTE: You should configure ./chat-ui/.env.local first before running this script."

pushd chat-ui

docker build \
    -t chat-ui \
    --secret id=DOTENV_LOCAL,src=.env.local \
    .

docker run \
    --rm \
    --env PORT=80 \
    --name chat-ui \
    --network host \
    chat-ui

popd
