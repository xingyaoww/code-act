from datasets import Dataset, DatasetDict
import pandas as pd

# Load the data into a pandas dataframe.
CODEACT_INSTRUCT = [
    "data/datasets/oct28_full6728.jsonl",
    "data/datasets/nov2_gpt4hard411.jsonl"
]

GENERAL_INSTRUCT = [
    "data/datasets/sharegpt_gpt4_all.jsonl",
    "data/datasets/sharegpt.n10000.jsonl",
    "data/datasets/openorca.n50000.jsonl",
    "data/datasets/capybara.jsonl"
]

def load_data(paths):
    df = []
    for path in paths:
        name = path.split("/")[-1].replace(".jsonl", "")
        _df = pd.read_json(path, lines=True)
        _df["id"] = _df["id"].apply(lambda x: f"{name}/{x}")
        df.append(_df)
    return pd.concat(df)

codeact_df = load_data(CODEACT_INSTRUCT)
general_df = load_data(GENERAL_INSTRUCT)

# Create a dataset from the pandas dataframe.
codeact_dataset = Dataset.from_pandas(codeact_df, preserve_index=False).shuffle()
general_dataset = Dataset.from_pandas(general_df, preserve_index=False).shuffle()

ds = DatasetDict({
    "codeact": codeact_dataset,
    "general": general_dataset
})
ds.push_to_hub("xingyaoww/code-act")
