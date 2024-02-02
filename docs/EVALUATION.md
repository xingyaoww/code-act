# :bar_chart: Evaluation 

## Setup

Make sure you've cloned all the submodules of this repo:

```bash
# Clone Repository and Submodules
git clone https://github.com/xingyaoww/code-act
git submodule update --init --recursive

# Make sure you already setup the environment
conda env create -f environment.yml
```

## M<sup>3</sup>ToolEval for comparing Code/Text/JSON

You should host the LLM you want to an OpenAI-Compatble API using e.g. [using vLLM](https://docs.vllm.ai/en/latest/).
Then you should add your model to the [`scripts/eval/m3tooleval/run.sh`](../scripts/eval/m3tooleval/run.sh) and direct run it as follows:

```bash
conda activate code-act
pushd scripts/eval/m3tooleval
./run.sh
```


## API-Bank

Similar to ActEval, you need to host your model into OpenAI-compatible API and modify the [`scripts/eval/api-bank/run.sh`](../scripts/eval/api-bank/run.sh). Afterwards, you can run the evaluation as follows:

```bash
# Setup depnendency
conda activate code-act
pushd scripts/eval/api-bank
# Install dependency for API-bank if you haven't
pip install -r requirements.txt
./run.sh
```

## Run Other Benchmarks

You should first setup data and install dependencies:

```bash
# Setup environment
./scripts/eval/setup_env.sh

# Setup evaluation data
./scripts/eval/setup_data.sh
```

**Easy way to run everything:** You do NOT need to serve the model! The script will automatically starts an vLLM server in a tmux session (you can check via `tmux ls`), and start every evaluation in parallel. You should specify `CUDA_VISIBLE_DEVICES` to set the GPUs you'd like to use!

The tmux session will be automatically killed when every evaluation is done (`$DIR_TO_HF_CHECKPOINT/eval/TASK_NAME/DONE` will be created when eval for that task is done).

NOTE: Before running everything, you should set your `OPENAI_API_KEY` as environment variable since MT-Bench requires GPT-4 for evaluation. You can mannully comment out corresponding lines in `scripts/eval/mint-bench/mint.sh` to disable it.

```bash
# Run eval on huggingface checkpoint
# the results will be saved under $DIR_TO_HF_CHECKPOINT/eval
./scripts/eval/run_all.sh $DIR_TO_HF_CHECKPOINT
```

When done, the evaluation results will be automatically aggregated into one JSON file and copied to `data/results/YOUR_EXP_ID/model_iter_YOUR_ITER.json`. If this is not happening, you can manually aggregate results by running:

```bash
python3 scripts/eval/aggregate_eval.py $DIR_TO_HF_CHECKPOINT
```

**For SLURM user:** We provide an example slurm script for you to run evaluation on your cluster, you can simply run:

```bash
# Make sure to update the slrum script with your cluster configuration!
sbatch scripts/slurm/configs/eval_2xA100.slurm $DIR_TO_HF_CHECKPOINT
```

**Separate Evaluation:** If you want run evaluation separately for each task, check the following sections. NOTE you need to start an vLLM server (with `--serve-model-name code-act-agent`) beforehand for these separate evaluation.


### MINT-Bench

```bash
./scripts/eval/mint-bench/mint.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/mint-bench`
```

### MiniWob++

```bash
./scripts/eval/miniwob++/miniwob++.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/miniwob++`
```

### ScienceWorld

```bash
./scripts/eval/science-world/science-world.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/science-world`
```

### GSM-8K

```bash
./scripts/eval/gsm8k/gsm8k.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/gsm8k`
```

### HumanEval

```bash
./scripts/eval/human_eval/human_eval.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/human_eval`
```

### MMLU

```bash
./scripts/eval/mmlu/mmlu.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/mmlu`
```


### MT-Bench

```bash
./scripts/eval/mt-bench/mt-bench.sh $DIR_TO_HF_CHECKPOINT
# results will be saved to `$DIR_TO_HF_CHECKPOINT/eval/mt-bench`
```

