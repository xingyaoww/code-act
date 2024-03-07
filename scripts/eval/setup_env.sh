#!/bin/bash

source scripts/eval/source.sh
check_conda_env_and_activate code-act
# if jq is not installed, install it
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Installing jq..."
    conda install -c conda-forge jq -y
fi

pip install vllm

# ==== Setup for MT-Bench ====
pushd scripts/eval/mt-bench/FastChat
pip install -e ".[model_worker,webui,llm_judge]"
popd

# ==== Setup for HumanEval ====
pushd scripts/eval/human_eval/human-eval
pip install -e .
popd

# ==== Setup for MiniWob++ ====
pip3 install gym selenium
pushd scripts/eval/miniwob++
pip install -e computergym
popd

# ==== Setup for Science World ====
pip3 install scienceworld==1.1.3 editdistance
pip3 install --no-deps sentence-transformers
pip install py4j
conda install -c conda-forge openjdk -y

