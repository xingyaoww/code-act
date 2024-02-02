import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input-filepath", type=str, default="data/raw/hotpotqa/train.json", help="Glob pattern for input files")
parser.add_argument("--output-filepath", type=str, default="data/processed/reasoning/train/hotpotqa.jsonl", help="Output file")
parser.add_argument("--keep-levels", default="hard", help="The difficulties to include")
parser.add_argument("--target-amt", type=int, default=3000, help="The number of examples to keep")
args = parser.parse_args()

with open(args.input_filepath, "r") as file:
    raw_data = json.load(file)
print(f"Loaded {len(raw_data)} examples")

data = []
keep_levels = args.keep_levels.split(",")
for idx, i in enumerate(raw_data):
    if i["level"] in keep_levels:
        data.append({"id": idx, "prompt": i["question"], "reference": i["answer"]})
print(f"Kept {len(data)} examples with levels: {keep_levels}")

if not os.path.exists(os.path.dirname(args.output_filepath)):
    os.makedirs(os.path.dirname(args.output_filepath))

df = pd.DataFrame(data)

if len(df) > args.target_amt:
    print(f"Downsampled to {args.target_amt} examples from {len(df)} examples")
    df = df.sample(args.target_amt, random_state=42)

df.to_json(args.output_filepath, orient="records", lines=True)

# Loaded 90447 examples
# Kept 15661 examples with levels: ['hard']
# Downsampled to 3000 examples from 15661 examples
