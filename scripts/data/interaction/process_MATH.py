"""Merge preprocessed data (from CRAFT) with original data (from MATH) and filter by difficulty."""
import re
import json
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob', type=str, default='data/raw/math/MATH/{split}/**/*.json', help='Glob pattern for input files')
parser.add_argument('--preprocessed_glob', type=str, default='data/raw/math/preprocessed-CRAFT/{split}/*.jsonl', help='Glob pattern for preprocessed files')
parser.add_argument('--output_dir', type=str, default='data/processed/reasoning/{split}', help='Output file')
parser.add_argument('--lowest-difficulty', type=str, default='3', help='The lowest difficulty to include')
args = parser.parse_args()

global_example_id = 0

for split in ["train"]: #, "test"]:
    print(f"Processing {split} split ...")
    output_dir = args.output_dir.format(split=split)
    shutil.rmtree(output_dir, ignore_errors=True)
    input_glob = args.input_glob.format(split=split)
    preprocessed_glob = args.preprocessed_glob.format(split=split)
    print(f"Output dir: {output_dir}")
    # make sure output dir exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading Raw Data ...")
    PROBLEM_TO_SOLUTION = {}
    input_files = glob(input_glob)
    for input_file in tqdm(input_files):
        with open(input_file) as f_in:
            data = json.load(f_in)
            assert "problem" in data
            assert "level" in data
            assert "type" in data
            assert "solution" in data
            PROBLEM_TO_SOLUTION[data["problem"]] = data["solution"]

    print("Loading Preprocessed Data ... Adding Solutions field from Raw Data ...")
    preprocessed_files = glob(preprocessed_glob)
    for preprocessed_file in tqdm(preprocessed_files):
        filename = pathlib.Path(preprocessed_file).name
        cur_file = []
        with open(preprocessed_file) as f_in:
            for line in f_in:
                data = json.loads(line)
                data["id"] = global_example_id
                global_example_id += 1
                data["extra"] = {
                    "raw_solution": PROBLEM_TO_SOLUTION[data["question"]]
                }
                cur_file.append(data)
        df = pd.DataFrame(cur_file)

        # filter by difficulty
        if split == "train":
            # level is a string like "Level 1"
            _pre_filter_len = len(df)
            # import pdb; pdb.set_trace()
            df = df[
                df["level"].apply(
                    lambda x: any(str(i) in x for i in range(1, 6)) and # filter out "Level ?"
                    re.search(r"Level (\d+)", x).group(1)
                ).astype(int) >= int(args.lowest_difficulty)
            ]
            print(f"Filtered {split} from {_pre_filter_len} to {len(df)} based on difficulty")

        output_filepath = pathlib.Path(output_dir) / filename
        df.to_json(output_filepath, orient="records", lines=True)
        print(f"Saved to {output_filepath}")


"""
Processing train split ...
Output dir: data/processed/reasoning/train
Loading Raw Data ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7500/7500 [00:02<00:00, 2613.99it/s]
Loading Preprocessed Data ... Adding Solutions field from Raw Data ...
  0%|                                                                                                                                                                                             | 0/7 [00:00<?, ?it/s]Filtered train from 870 to 727 based on difficulty
Saved to data/processed/reasoning/train/geometry.jsonl
Filtered train from 1205 to 750 based on difficulty
Saved to data/processed/reasoning/train/prealgebra.jsonl
Filtered train from 771 to 602 based on difficulty
Saved to data/processed/reasoning/train/counting_and_probability.jsonl
Filtered train from 1295 to 1070 based on difficulty
Saved to data/processed/reasoning/train/intermediate_algebra.jsonl
 57%|███████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                             | 4/7 [00:00<00:00, 38.36it/s]Filtered train from 869 to 691 based on difficulty
Saved to data/processed/reasoning/train/number_theory.jsonl
Filtered train from 746 to 520 based on difficulty
Saved to data/processed/reasoning/train/precalculus.jsonl
Filtered train from 1744 to 1226 based on difficulty
Saved to data/processed/reasoning/train/algebra.jsonl
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 42.67it/s]
"""
