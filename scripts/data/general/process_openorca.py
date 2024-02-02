import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data/datasets")
parser.add_argument("--subsample_size", type=int, default=None)
args = parser.parse_args()

dataset = load_dataset("Open-Orca/OpenOrca", split="train")
# to dataframe
df = dataset.to_pandas()
print(f"Loaded {len(df)} examples")

def normalize_conv(row):
    return {
        "id": row["id"],
        "conversations": [
            {
                "role": "system",
                "content": row["system_prompt"]
            },
            {
                "role": "user",
                "content": row["question"]
            },
            {
                "role": "assistant",
                "content": row["response"]
            }
        ]
    }


print(f"Before filtering: {len(df)} examples")
df = df[df["id"].str.startswith("cot.")]
print(f"After filtering (keep CoT): {len(df)} examples")

print(f"Normalizing {len(df)} examples")
df = df.progress_apply(normalize_conv, axis=1, result_type="expand")

if args.subsample_size is not None:
    output_filepath = os.path.join(args.output_dir, f"openorca.n{args.subsample_size}.jsonl")
    df = df.sample(args.subsample_size, random_state=42, replace=False)
    print(f"After Subsample: {len(df)} examples")
else:
    output_filepath = os.path.join(args.output_dir, "openorca.jsonl")

print(f"Writing {len(df)} examples to {output_filepath}")
df.to_json(output_filepath, orient="records", lines=True)
