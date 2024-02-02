"""Merge preprocessed data (from CRAFT) with original data (from MATH)."""
import json
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/raw/apps/{split}.jsonl', help='Glob pattern for input files')
parser.add_argument('--output_dir', type=str, default='data/processed/code_generation/{split}', help='Output file')
parser.add_argument('--difficulties', type=str, default='introductory,interview,competition', help='The difficulties to include')
args = parser.parse_args()

def generate_prompt(sample) -> str:
    # https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/example_script.py#L12
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"] 
    except ValueError:
        fn_name = None 
    
    _input = ""
    _input += sample["question"] + "\n"
    if starter_code:
        _input += starter_code
    if fn_name:
        # _input += "\nUse Standard Input format"
        _input += "\nYou should complete the provided starter code by filling in the function body for " + fn_name + "."
    else:
        # _input += "\nUse Call-Based format"
        _input += "\nYou should write code that expect inputs from stdin and print outputs (to stdout)."
    
    # _input += "\nANSWER:\n"
    return _input

for split in ["train"]: #, "test"]:
    print(f"Processing {split} split ...")
    output_dir = args.output_dir.format(split=split)
    shutil.rmtree(output_dir, ignore_errors=True)
    input_path = args.input_path.format(split=split)
    print(f"Output dir: {output_dir}")
    # make sure output dir exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_json(input_path, orient="records", lines=True)
    print("Loading Raw Data with {} examples".format(len(df)))
    df["prompt"] = df.apply(generate_prompt, axis=1)
    # rename input_output to reference
    df = df.rename(columns={"input_output": "reference"})
    # remove instances with empty reference
    df = df[df["reference"].apply(lambda x: len(x) > 0)]
    print(f"Filtered to {len(df)} by removing empty reference")

    if split == "train":
        # select difficult examples
        selected_difficulties = args.difficulties.split(",")
        _pre_filter_len = len(df)
        df = df[df["difficulty"].isin(selected_difficulties)]
        print(f"Filtered {len(df)} out of {_pre_filter_len} examples with difficulties: {selected_difficulties}")

        # take first 10 examples as few-shot
        df_few_shot = df.iloc[:10]
        df = df.iloc[10:]
        df_few_shot.to_json(pathlib.Path(output_dir) / "apps.fewshot.jsonl", orient="records", lines=True)

    # remove instances with empty reference
    def _has_test_case(x):
        try:
            x = json.loads(x)
        except ValueError:
            print(f"Error load test case -- skipping...")
            return False
        if len(x.get("inputs", [])) == 0:
            return False
        return True
    df = df[df["reference"].apply(_has_test_case)]
    print(f"Filtered to {len(df)} by removing empty reference (0 test case)")
    
    df.to_json(pathlib.Path(output_dir) / "apps.jsonl", orient="records", lines=True)

"""
Processing train split ...
Output dir: data/processed/code_generation/train
Loading Raw Data with 5000 examples
Filtered to 4805 by removing empty reference
Filtered 4805 out of 4805 examples with difficulties: ['introductory', 'interview', 'competition']
Error load test case -- skipping...
Filtered to 4439 by removing empty reference (0 test case)
"""
