# ScienceWorld

This is evaluation code is modified from [AgentTuning](https://github.com/THUDM/AgentTuning/tree/main).

---

This evaluation code is adapted from [SwiftSage](https://github.com/yuchenlin/SwiftSage).

## Installation

```bash
conda create -n sciworld python=3.8 pip
conda activate sciworld
pip3 install scienceworld==1.1.3
pip3 install -r requirements.txt
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install -c conda-forge openjdk # if needed 
```

## Evaluation

### GPT

Modify `eval-gpt.sh` to provide your OpenAI API key and optionally specify a model. Then run:

```bash
bash eval-gpt.sh
```

### HuggingFace TGI (Text Generation Inference)

Modify `eval-tgi.sh` to provide your TGI controller addresses in an comma-separated array. Then run:

```bash
bash eval-tgi.sh
```

----

After evaluation is done, you can run `python metrics.py` to get the results.
