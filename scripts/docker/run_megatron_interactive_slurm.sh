#!/bin/bash

WORK_DIR=`pwd`
IMAGE=TODO_YOUR_IMAGE_DIR/pt-megatron-llm_v1.1.1.sif
# only for verify correctness - since NCSA cuda driver version does not fully support it yet
echo "WORK_DIR=$WORK_DIR"
echo "IMAGE=$IMAGE"

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata
module load cuda/12.2.1

apptainer run --nv \
    --no-home \
    --no-mount bind-paths \
    --cleanenv \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --writable-tmpfs \
    --bind /scratch:/scratch \
    --bind $WORK_DIR:/workspace \
    $IMAGE \
    /bin/bash

