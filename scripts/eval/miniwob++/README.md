# MiniWoB++

This is evaluation framework is modified from [AgentTuning](https://github.com/THUDM/AgentTuning/tree/main).

---

We take the framework of [RCI Agent](https://github.com/posgnu/rci-agent) to evaluate the performance of our model on MiniWoB++.

## Installation

```bash
pip install -r requirements.txt
pip install -e computergym
```

## Evaluation

### GPT

```bash
bash eval-gpt3.5.sh
```

or 

```bash
bash eval-gpt4.sh
```

You need to provide your OpenAI API key as an environment variable `OPENAI_API_KEY`.

### TGI (Text Generation Inference)

```bash
bash eval-tgi.sh
```

You need to modify `eval-tgi.sh` to provide your TGI controller addresses in an comma-separated array.
