# 📂 Data Generation

This docs describes how to download raw data, generate interaction trajectories, and filter and post-processing them for model training. Please refer to the paper for more details.


## Setup

```bash
# Clone Repository and Submodules
git clone https://github.com/xingyaoww/code-act
git submodule update --init --recursive

# Setup environment
conda env create -f environment.yml
```

## Dataset Preparation

```bash
# Download raw dataset (it will download data for differet tasks to `data/raw`)
scripts/data/interaction/download_raw_data.sh
# Process datasets 
python3 scripts/data/interaction/process_APPS.py
python3 scripts/data/interaction/process_hotpotQA.py
python3 scripts/data/interaction/process_MATH.py
python3 scripts/data/interaction/process_WikiTableQuestions.py
```

## Start dataset collection

CodeAct uses MINT's framework for trajectory generation (`mint/`). You need to be able to get into the docker environment for MINT's code execution OR setup a conda environment following MINT's repo [https://github.com/xingyaoww/mint-bench](https://github.com/xingyaoww/mint-bench).

```bash
# Use docker env (please check https://github.com/xingyaoww/mint-bench for conda instruction)
./scripts/data/interaction/collection/run_mint_docker_interactive.sh

# Once in the docker environment
./scripts/data/interaction/collection/run.sh
```

`scripts/data/interaction/collection/run.sh` uses configuration files defined in [`mint/data_gen_configs`](../mint/data_gen_configs). You can refer to [this](https://github.com/xingyaoww/mint-bench?tab=readme-ov-file#generate-config) for more information about config files. You can change the config file to suit your need.


## Post-process the generated trajecories

The generated trajectories will be saved to `data/outputs/<MODEL_NAME>` by default. Please refer to `scripts/data/interaction/collection/convert_outputs.ipynb` about how to convert outputs into trajectories. Once done, you will have a folder under `data/trajectories/` for the next step.

You can *optionally* running these post-processed trajectories (e.g., harder instances) on GPT-4: you can put the trajectory directory in the corresponding data generation configuration file (e.g., [`mint/data_gen_configs/gpt-4-0613/F=None/max5_p2+tool+cd/code_generation/APPS/train.json`](../mint/data_gen_configs/gpt-4-0613/F=None/max5_p2+tool+cd/code_generation/APPS/train.json)) for running the next data generation task.

## Process Trajectories into formats ready for training

Use [`scripts/data/interaction/collection/process_trajectories.ipynb`](../scripts/data/interaction/collection/process_trajectories.ipynb) by putting `data/trajectories/*` folder as input to process your generated data into instances (e.g., `data/datasets/nov2_gpt4hard411.jsonl`) ready for training.

Each line of the output is in the format of:

```json
{
    "id": "XXX",
    "conversations": [
        {"role": "system", "content": "XXX"},
        {"role": "user", "content": "XXX"},
        {"role": "assistant", "content": "XXX"}
        // ...
    ]
}
```

Please refer to [`docs/MODEL_TRAINING.md`](./MODEL_TRAINING.md) for details of model training.

## Additional Tool

- `scripts/data/analyze_dataset.ipynb`: You can use this notebook to analyze the number of tokens per dataset.
