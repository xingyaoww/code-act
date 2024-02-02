#!/bin/bash
DOCKER_IMG=xingyaoww/pt-megatron-llm:v1.0

WORK_DIR=`pwd`

docker run \
    -e UID=$(id -u) \
    -e WANDB_API_KEY \
    -e HUGGING_FACE_HUB_TOKEN \
    --gpus all -it \
    --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $WORK_DIR:/workspace \
    -v /data/shared:/models \
    $DOCKER_IMG \
    bash -c "useradd --shell /bin/bash -u $UID -o -c '' -m code-act-agent && cd /workspace && su code-act-agent -c 'git config --global credential.helper store' && su code-act-agent"
