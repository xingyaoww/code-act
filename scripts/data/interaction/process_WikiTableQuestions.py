import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--input-filepath", type=str, default="data/raw/wiki_table_question/WikiTableQuestions/data/training.tsv")
parser.add_argument("--table-glob", type=str, default="data/raw/wiki_table_question/WikiTableQuestions/csv/**/*.csv")
parser.add_argument("--output-filepath", type=str, default="data/processed/tabular/train/wiki_table_questions.jsonl", help="Output file")
parser.add_argument("--target-amt", type=int, default=3000, help="The number of examples to keep")
args = parser.parse_args()

pathlib.Path(args.output_filepath).parent.mkdir(parents=True, exist_ok=True)

# Load the table data
table_paths = glob(args.table_glob, recursive=True)
print(f"Found {len(table_paths)} tables.")

def read_table(table_path):
    try:
        # First, try reading as CSV
        assert table_path.endswith(".csv")
        table = pd.read_csv(table_path)
    except Exception:
        # If CSV reading fails, attempt reading as TSV
        try:
            # Change the file extension to .csv and read as TSV
            csv_table_path = table_path.replace('.csv', '.tsv')
            table = pd.read_csv(csv_table_path, sep="\t")
        except Exception as e:
            print(f"{table_id} is not a valid table.")
            raise e
    return table

# Initialize the list to store table data
table_df = []
table_id_to_table = {}
for table_path in tqdm(table_paths):
    table_id = "/".join(table_path.split("/")[-2:]).rstrip(".csv")
    table = read_table(table_path)

    # Append the table details to the list
    table_df.append({
        "table_id": table_id,
        "num_rows": table.shape[0],
        "num_cols": table.shape[1],
    })
    table_id_to_table[table_id] = table

# Convert the list to DataFrame and display
table_df = pd.DataFrame(table_df)
print(f"Loaded {len(table_df)} tables.")

# Load Training Data
train_ds = pd.read_csv(args.input_filepath, sep="\t")
print(f"Found {train_ds.shape[0]} training examples.")
# match example with table stats
train_ds["table_id"] = train_ds["context"].apply(lambda x: x.lstrip("csv/").rstrip(".csv"))
train_ds = train_ds.merge(table_df, on="table_id", how="left")

def process_ds(train_ds):

    train_ds = train_ds.assign(table=train_ds["table_id"].apply(lambda x: table_id_to_table[x]))

    # Rename to unified format
    train_ds = train_ds.rename(columns={"utterance": "prompt", "targetValue": "reference"})

    # We have two format of question: SQL and Pandas; Assign them randomly
    # set random seed for reproducibility
    train_ds = train_ds.sample(frac=1, random_state=42).reset_index(drop=True)
    # assign half to SQL and half to Pandas
    train_ds["question_format"] = np.where(train_ds.index < len(train_ds) / 2, "sql", "pandas")
    print(f"Assigned {sum(train_ds['question_format'] == 'sql')} examples to SQL format and {sum(train_ds['question_format'] == 'pandas')} examples to Pandas format.")

    # shuffle the dataset again
    train_ds = train_ds.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add instruction:
    def format_instruction(row):
        table_str = str(row["table"].head())

        if row["question_format"] == "sql":
            return (
                f"Given the following table (only the first 5 rows are shown):\n"
                f"{table_str}\n\n"
                f"Write a SQL query to find the answer to the question: {row['prompt']}.\n"
                f"The SQLite3 database is preloaded for you and can be accessed within <execute> block via the variable `conn` (SQLite3 connection object).\n"
                f"The table name is `data_table`.\n"
            )
        elif row["question_format"] == "pandas":
            return (
                f"Given the following table (only the first 5 rows are shown):\n"
                f"{table_str}\n\n"
                f"Write a Pandas query to find the answer to the question: {row['prompt']}.\n"
                f"The dataframe is preloaded for you and can be accessed within <execute> block via the variable `df`.\n"
            )
        else:
            raise ValueError("Invalid question format.")

    train_ds["prompt"] = train_ds.apply(format_instruction, axis=1)
    return train_ds

# Pick the top target_amt examples with the most amount of rows and columns
print(f"Downsampled to {args.target_amt} examples from {len(train_ds)} examples based on table size.")
sorted_train_ds = train_ds.sort_values(["num_rows", "num_cols"], ascending=False)

train_ds = sorted_train_ds.head(args.target_amt)
train_ds = process_ds(train_ds)
train_ds.to_json(args.output_filepath, orient="records", lines=True)

# sample 50 examples from the end of sorted_train_ds for in-context example selection
in_context_ds = sorted_train_ds.tail(50)
in_context_ds = process_ds(in_context_ds)
with open(args.output_filepath.replace(".jsonl", ".icl.txt"), "w") as f:
    for row in in_context_ds.itertuples():
        f.write("====================")
        f.write("Table ID: " + row.table_id)
        f.write("\n\n")
        f.write(row.prompt)
        f.write("\n\n")
        f.write("Answer:" + row.reference)
        f.write("\n\n")

"""
Loaded 2108 tables.
Found 14149 training examples.
Downsampled to 3000 examples from 14149 examples based on table size.
Assigned 1500 examples to SQL format and 1500 examples to Pandas format.
"""
