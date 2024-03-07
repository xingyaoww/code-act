#!/bin/bash

source scripts/eval/source.sh

# MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P data/eval/mmlu
tar -xvf data/eval/mmlu/data.tar -C data/eval/mmlu
rm data/eval/mmlu/data.tar
mv data/eval/mmlu/data/* data/eval/mmlu
rm -r data/eval/mmlu/data

# MATH
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar -P data/eval/math
tar -xvf data/eval/math/MATH.tar -C data/eval/math
rm data/eval/math/MATH.tar
mv data/eval/math/MATH/* data/eval/math
rm -r data/eval/math/MATH

# GSM8K
check_conda_env_and_activate code-act
python3 -c "import datasets; dataset = datasets.load_dataset('gsm8k', 'main'); dataset.save_to_disk('data/eval/gsm8k')"
