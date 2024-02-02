#!/bin/bash

# Single-Turn Chain-of-Thought
# - OpenOrca (system, user, assistant)
# Originally 141695 CoT open-orca trajs.

python3 scripts/data/general/process_openorca.py \
    --output_dir data/datasets \
    --subsample_size 50000
# Loaded 4233923 examples
# Before filtering: 4233923 examples
# After filtering (keep CoT): 141695 examples
# Normalizing 141695 examples
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 141695/141695 [00:07<00:00, 18997.38it/s]
# After Subsample: 50000 examples
# Writing 50000 examples to data/datasets/openorca.n50000.jsonl

# Multi-Turn Chat
# - ShareGPT
mkdir -p data/raw
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json -O data/raw/sharegpt.json


python3 scripts/data/general/process_sharegpt.py \
    --input_file data/raw/sharegpt.json \
    --output_dir data/datasets \
    --min_turns 4 \
    --subsample_size 10000

# Loaded 94145 examples
# Role count (once per turn): Counter({'gpt': 93256, 'human': 92592, 'system': 212, 'chatgpt': 33, 'bing': 23, 'user': 13, 'bard': 6})
# Dataset size before filtering: 94145
# Dataset size after filtering (keep turn count >= 4): 69048
# Dataset size after filtering (has at least 1 user turn): 69029
# Dataset size after filtering (has at least 1 assistant turn): 69012
# Dataset size after filtering (has no turns exactly match 'Cancelled'): 69011
# Dataset size after filtering (first turn is not assistant): 39614
# Dataset size after filtering (no conv with consecutive assistant turns): 39612
# Dataset size after filtering (no conv with consecutive user turns): 39537
# Role count (once per turn, after normalization & filtering): Counter({'user': 39537, 'assistant': 39537})
# After Subsample: 10000 examples

wget https://huggingface.co/datasets/openchat/openchat_sharegpt_v3/resolve/main/sharegpt_gpt4.json?download=true -O data/raw/sharegpt_gpt4.json

python3 scripts/data/general/process_sharegpt.py \
    --input_file data/raw/sharegpt_gpt4.json \
    --output_dir data/datasets \
    --min_turns 2 \
    --output_filename sharegpt_gpt4_all

# Loaded 6599 examples
# Rename 'items' to 'conversations'
# Role count (once per turn): Counter({'gpt': 6599, 'human': 6599})
# Dataset size before filtering: 6599
# Dataset size after filtering (keep turn count >= 2): 6599
# Dataset size after filtering (has at least 1 user turn): 6599
# Dataset size after filtering (has at least 1 assistant turn): 6599
# Dataset size after filtering (has no turns exactly match 'Cancelled'): 6599
# Dataset size after filtering (first turn is not assistant): 6599
# Dataset size after filtering (no conv with consecutive assistant turns): 6599
# Dataset size after filtering (no conv with consecutive user turns): 6599
# Role count (once per turn, after normalization & filtering): Counter({'user': 6599, 'assistant': 6599})
# Writing 6599 examples to data/datasets/sharegpt_gpt4_all.jsonl


python3 scripts/data/general/process_capybara.py \
    --output_dir data/datasets
# Generating train split: 663 examples [00:01, 455.93 examples/s]
# Loaded 4647 examples
# Normalizing 4647 examples
# 100%|███████████████████████████████████████████████████████████████| 4647/4647 [00:00<00:00, 19524.90it/s]
# Writing 4647 examples to data/datasets/capybara.jsonl
