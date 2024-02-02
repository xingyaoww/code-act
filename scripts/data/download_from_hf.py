import os
import pathlib
from datasets import load_dataset

ds = load_dataset("xingyaoww/code-act")

if not os.path.exists("data/datasets"):
    pathlib.Path("data/datasets").mkdir(parents=True, exist_ok=True)
    print("Created data/datasets")

codeact_ds = ds["codeact"]
codeact_df = codeact_ds.to_pandas()
codeact_df.to_json("data/datasets/codeact.jsonl", orient="records", lines=True)
print(f"Saved {len(codeact_df)} examples to data/datasets/codeact.jsonl")

general_ds = ds["general"]
general_df = general_ds.to_pandas()
general_df.to_json("data/datasets/general.jsonl", orient="records", lines=True)
print(f"Saved {len(general_df)} examples to data/datasets/general.jsonl")
print("Done")
