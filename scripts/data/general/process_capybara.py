import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data/datasets")
args = parser.parse_args()

# Used for https://huggingface.co/LDJnr/Capybara-V1.9-GGUF
dataset_1 = load_dataset("LDJnr/Verified-Camel", split="train")
dataset_2 = load_dataset("LDJnr/Pure-Dove", split="train")
dataset_3 = load_dataset("LDJnr/LessWrong-Amplify-Instruct", split="train")

dataset_1 = dataset_1.to_pandas()
dataset_1["id"] = dataset_1["source"] + "." + dataset_1.index.astype(str)
dataset_2 = dataset_2.to_pandas()
dataset_2["id"] = dataset_2["source"] + "." + dataset_2.index.astype(str)
dataset_3 = dataset_3.to_pandas()
dataset_3["id"] = dataset_3["source"] + "." + dataset_3.index.astype(str)

# to dataframe
df = pd.concat([dataset_1, dataset_2, dataset_3])
df = df[["id", "conversation"]]
print(f"Loaded {len(df)} examples")

def normalize_conv(row):
    conversations = []
    for turn in row["conversation"]:
        _inp = turn["input"]
        _out = turn["output"]
        conversations.append({
            "role": "user",
            "content": _inp
        })
        conversations.append({
            "role": "assistant",
            "content": _out
        })
    return {
        "id": row["id"],
        "conversations": conversations
    }


print(f"Normalizing {len(df)} examples")
df = df.progress_apply(normalize_conv, axis=1, result_type="expand")

output_filepath = os.path.join(args.output_dir, "capybara.jsonl")

print(f"Writing {len(df)} examples to {output_filepath}")
df.to_json(output_filepath, orient="records", lines=True)
