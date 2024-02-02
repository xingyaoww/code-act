# :blue_book: Model Training

We use a [fork of Megatron-LLM for training the models](https://github.com/xingyaoww/Megatron-LLM) for efficiency.

The typical training throughput for `Mistral-7b-v0.1` on one 4xA100 SXM (40G) node is around 9k tokens per second with our default configuration file and takes around ~10 hour to complete training model for 5 epochs.


## Data Preparation

**The easy way:** You can simply download our already processed data set from [huggingface dataset 🤗](https://huggingface.co/datasets/xingyaoww/code-act).

```bash
python3 scripts/data/download_from_hf.py
# it will download dataset from huggingface:
# Saved 7139 examples to data/datasets/codeact.jsonl
# Saved 71246 examples to data/datasets/general.jsonl
```

**For reproducibility:** To fully reproduce CodeActAgent, you can follow [docs/DATA_GENERATION.md](./DATA_GENERATION.md) to generate your CodeAct multi-turn interaction data, and use [`scripts/data/general/process_general_traj.sh`](../scripts/data/general/process_general_traj.sh) to download and process conversation data for training.


## Environment Setup

We recommend directly using the [docker image we provided](../scripts/docker/Dockerfile.megatron) based on nvcr docker image from nvidia.

**Docker:** If you have access to docker on your machine, you can run [`./scripts/docker/run_megatron_interactive.sh`](../scripts/docker/run_megatron_interactive.sh) directly to get into the container environment.

**Apptainer:** If you are using SLURM cluster that support [apptainer](https://apptainer.org/) (former Singularity), you can directly pull this docker image into apptainer format and run on your SLURM cluster:

```bash
apptainer pull pt-megatron-llm_v1.1.1.sif docker://xingyaoww/pt-megatron-llm:v1.1.1
# it will create an image `pt-megatron-llm_v1.1.1.sif` on your current working directory
```

Then you can modify [`scripts/docker/run_megatron_interactive_slurm.sh`](../scripts/docker/run_megatron_interactive_slurm.sh) to put in the directory of your image file. Once done, you can run `scripts/docker/run_megatron_interactive_slurm.sh` to get into the apptainer environment for interactive debugging / data processing.

We provide example slurm script (`srun`) for you at `scripts/slurm/run_megatron_gpu_interactive.sh`. You can modify it based on your cluser configuration.


## Model Conversion

Megatron-LLM requires a specific format of model to work. You should already entered an interactive session of the Megatron-LLM environment following the "Environment Setup" section above.

**Convert Huggingface Model to Megatron:** We currently have script for converting LLaMA-2 and Mistral model from huggingface. You should first download these models (using huggingface's `git clone`) to `data/models/raw_hf`:

```bash
data/models/raw_hf/
├── Llama-2-7b-hf
└── Mistral-7B-v0.1
```

Then you can run
```bash
./scripts/models/megatron/convert_llama.sh
# OR
./scripts/models/megatron/convert_mistral.sh
```

The model will then be converted to `data/models/raw/Llama-2-7b-megatron` or `data/models/raw/Mistral-7b-megatron` for further use.

**Shard Megatron model for training:**
If you are training model using multiple GPUs, you need to shard the model weight. For example, if you are training on 4xA100, you can shard to 4 partitions of tensor parallel.
You can do so by running:

```bash
./scripts/models/megatron/shard_model_4tp.sh LLama-2-7b
# OR
./scripts/models/megatron/shard_model_4tp.sh Mistral-7b
# OR you can modify the script to shard to your desired partition
```

The resulting model will be available at `data/models/sharded` (e.g., `data/models/sharded/Mistral-7b-megatron-tp4-pp1` for TP=4 and PP=1).

## Convert Data Format to Megatron-LLM

We need to convert dataset into a Megatron-specific format for training effeciency. 
The conversion script will handle tokenization and chat format ([ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) is used by default).
We will also perform dataset packing (i.e., packing multiple data instances into one sequence) for maxmial training effeciency.

You will see something like "# Packed 78385 documents into 19379 documents", which means you packed 78k training instances into 19k longer instances (each contains one or multiple original training instances). You should use the latter number (19379) for the length of dataset in the training script.

You need to look at the script and uncomment one line if you are not using the dataset downloaded from huggingface.

```bash
# For LLaMA model and tokenizer (pack to sequence length 4096)
./scripts/data/megatron_conversion/process_mixture_llama.sh
# For Mistral model and tokenizer (pack to sequence length 16k)
./scripts/data/megatron_conversion/process_mixture_mistral.sh
```

## Start model training!

If you are directly training the model on a **local/docker environment**:

```bash
./scripts/models/megatron/finetune_4xA100_4tp_mixture_llama.sh
# OR
./scripts/models/megatron/finetune_4xA100_4tp_mixture_mistral.sh
```

You can change configurations in the training script for your hardware. The example training script is optimized for 4xA100 40G SXM.

If you need to submit job to a **slurm cluster**, we provide an example slurm job script at `scripts/slurm/configs/finetune_4xA100_4tp.slurm`. Once modified to your cluster setting, you can launch it via:

```bash
# For LLaMA training
sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm scripts/models/megatron/finetune_4xA100_4tp_mixture_llama.sh
# OR
sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm scripts/models/megatron/finetune_4xA100_4tp_mixture_mistral.sh
```

The model checkpoints will be saved to `data/ckpts`.

## Convert back to huggingface format

When you done training, you will see a folder `data/ckpts/YOUR_EXP_NAME` that look like this:

```
.
├── iter_0000152
├── iter_0000304
├── iter_0000456
├── iter_0000608
├── iter_0000760
└── latest_checkpointed_iteration.txt
```

You can convert arbitrary checkpoint back to huggingface format by running:

```bash
# NOTE: YOUR_EXP_NAME need to contains either 'Llama-2' or 'Mistral-7b' otherwise it is not supported yet
# for example, to convert iter_0000152:
./scripts/models/megatron/convert_sharded_to_hf.sh data/ckpts/YOUR_EXP_NAME 152
```

The converted huggingface compatible model will be saved to `data/ckpts/YOUR_EXP_NAME/model_iter_152` with a `chat_template` ([ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)) automatically added for the tokenizer.
