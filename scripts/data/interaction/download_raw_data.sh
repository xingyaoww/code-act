# !/bin/bash

# APPS
mkdir -p data/raw/apps
wget https://huggingface.co/datasets/codeparrot/apps/resolve/main/train.jsonl -O data/raw/apps/train.jsonl
wget https://huggingface.co/datasets/codeparrot/apps/resolve/main/test.jsonl -O data/raw/apps/test.jsonl

# MATH
mkdir -p data/raw/math
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar -O data/raw/math/math.tar
tar -xvf data/raw/math/math.tar -C data/raw/math

mkdir -p data/raw/math/preprocessed-CRAFT/train
mkdir -p data/raw/math/preprocessed-CRAFT/test
SPLITS=("algebra" "counting_and_probability" "geometry" "intermediate_algebra" "number_theory" "prealgebra" "precalculus");
PREFIX_URL=https://raw.githubusercontent.com/lifan-yuan/CRAFT/main/tab_and_math/MATH/dataset
for SPLIT in "${SPLITS[@]}"; do
    wget ${PREFIX_URL}/${SPLIT}.jsonl -O data/raw/math/preprocessed-CRAFT/test/${SPLIT}.jsonl
    wget ${PREFIX_URL}/train/${SPLIT}.jsonl -O data/raw/math/preprocessed-CRAFT/train/${SPLIT}.jsonl
done

# AlfWorld
export ALFWORLD_DATA=data/raw/alfworld;
git clone https://github.com/xingyaoww/alfworld.git
./alfworld/scripts/alfworld-download
rm -rf alfworld

# HotpotQA
mkdir -p data/raw/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/raw/hotpotqa/train.json


# WikiTableQuestion
mkdir -p data/raw/wiki_table_question
wget "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip" -O data/raw/wiki_table_question/wiki_table_question.zip
unzip data/raw/wiki_table_question/wiki_table_question.zip -d data/raw/wiki_table_question
rm data/raw/wiki_table_question/wiki_table_question.zip
