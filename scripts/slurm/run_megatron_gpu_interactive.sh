#!/bin/bash
srun \
    -A YOUR_CLUSTER \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=16 \
    --tasks=1 \
    --cpus-per-task=16 \
    --partition=gpuA40x4 \
    --gpus=1 \
    --mem=60g \
    --pty scripts/docker/run_megatron_interactive_slurm.sh
