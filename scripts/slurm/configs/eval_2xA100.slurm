#!/bin/bash
#SBATCH --job-name="eval_2xA100"
#SBATCH --output="logs/eval/%j.%N.eval_2xA100.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=104G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch&projects"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=TODO_YOUR_ACCOUNT
#SBATCH --no-requeue
#SBATCH -t 4:00:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata
module load cuda/12.2.1

echo "job is starting on `hostname`"

# start a tmux session called eval
EVAL_CKPT=$1 # data/ckpts/Llama-2-7b-t4-p1-oct25_geninst_pack4096-lr1e-5-bs32-seq4096/hf/mint_agent_iter_875

cd `TODO_YOUR_DIR`
echo "EVAL_CKPT=$EVAL_CKPT"

scripts/eval/run_all.sh $EVAL_CKPT 1
