import os
import json
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file_glob", type=str, default="scripts/eval/mint-bench/mint-bench/data/outputs/code-act-agent/**/*.jsonl")
args = parser.parse_args()

files = glob(args.file_glob, recursive=True)
print(f"Found {len(files)} files")

for filepath in tqdm(files):
    task_id_set = set()

    old_count = 0
    new_outputs = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            old_count += 1
            task_id = d["task"]["task_id"]
            if task_id in task_id_set:
                continue
            task_id_set.add(task_id)
            new_outputs.append(d)
    
    # rename the file
    os.rename(filepath, filepath + ".dup_bak")

    with open(filepath, "w") as f:
        for d in new_outputs:
            f.write(json.dumps(d) + "\n")

    print(f"Removed {old_count - len(new_outputs)} duplicates from {filepath}")
