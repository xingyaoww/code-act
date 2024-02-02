import os
import json
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="data/raw/sharegpt.json")
parser.add_argument("--output_dir", type=str, default="data/datasets")
parser.add_argument("--output_filename", type=str, default="sharegpt")
parser.add_argument("--min_turns", type=int, default=4)
parser.add_argument("--subsample_size", type=int, default=None)
parser.add_argument("--english_only", action="store_true")
args = parser.parse_args()

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_json(args.input_file)
print(f"Loaded {len(df)} examples")

if "items" in df.columns:
    print(f"Rename 'items' to 'conversations'")
    df = df.rename(columns={"items": "conversations"})


ROLE_MAP = {
    # user
    "user": "user",
    "human": "user",

    # assistant
    "chatgpt": "assistant",
    "gpt": "assistant",
    "bard": "assistant",

    # system
    "system": "system"
}

def fix_artifact(text):
    # if "\\_" in text:
    #     import pdb; pdb.set_trace()
    text = text.replace("\\_", "_")
    return text

class UnrecognizedRoleError(Exception):
    pass

def normalize_conv_turn(conv_turn):
    role = conv_turn["from"]
    content = fix_artifact(conv_turn["value"])

    if role in ROLE_MAP:
        role = ROLE_MAP[role]
    elif role == "bing":
        role = "user"
        content = f"Bing Search Result:\n{content}"
    else:
        import pdb; pdb.set_trace()
        raise UnrecognizedRoleError(f"Unknown role: {role}")

    return {
        "role": role,
        "content": content
    }

def normalize_conv(conv):
    TEXTS_TO_FILTER = [
        "This chat conversation is shared from",
        "*This conversation is shared from",
    ]
    # remove turns that contain the key
    filtered_conv = []
    for turn in conv:
        if not any(key in turn["value"] for key in TEXTS_TO_FILTER):
            filtered_conv.append(normalize_conv_turn(turn))
    return filtered_conv

print(
    "Role count (once per turn):",
    df["conversations"]
    .apply(lambda x: set([turn["from"] for turn in x]))
    .apply(lambda x: Counter(x)).sum()
)

df["conversations"] = df["conversations"].apply(normalize_conv)


print(f"Dataset size before filtering: {len(df)}")
# Apply turn count filter
df = df[df["conversations"].apply(lambda x: len(x) >= args.min_turns)]
print(f"Dataset size after filtering (keep turn count >= {args.min_turns}): {len(df)}")

df = df[df["conversations"].apply(lambda x: any(turn["role"] == "user" for turn in x))]
print(f"Dataset size after filtering (has at least 1 user turn): {len(df)}")

df = df[df["conversations"].apply(lambda x: any(turn["role"] == "assistant" for turn in x))]
print(f"Dataset size after filtering (has at least 1 assistant turn): {len(df)}")

df = df[df["conversations"].apply(lambda x: not any("Cancelled" == turn["content"] for turn in x))]
print(f"Dataset size after filtering (has no turns exactly match 'Cancelled'): {len(df)}")

df = df[df["conversations"].apply(lambda x: x[0]["role"] != "assistant")]
print(f"Dataset size after filtering (first turn is not assistant): {len(df)}")

# find if there are consecutive assistant turns
df = df[df["conversations"].apply(
    lambda x: 
    not any(x[i]["role"] == "assistant" and x[i+1]["role"] == "assistant" for i in range(len(x)-1)))
]

if args.english_only:
    import cld3
    def is_english(text):
        detected_lang = cld3.get_language(text)
        if detected_lang.language == "en" and detected_lang.is_reliable:
            return True
        return False
    # Only the first user turn is in English
    df = df[df["conversations"].apply(lambda x: is_english(x[0]["content"]))]
    print(f"Dataset size after filtering (all turns are english): {len(df)}")

print(f"Dataset size after filtering (no conv with consecutive assistant turns): {len(df)}")

# find if there are consecutive user turns
df = df[df["conversations"].apply(
    lambda x:
    not any(x[i]["role"] == "user" and x[i+1]["role"] == "user" for i in range(len(x)-1)))
]
print(f"Dataset size after filtering (no conv with consecutive user turns): {len(df)}")

print(
    "Role count (once per turn, after normalization & filtering):",
    df["conversations"]
    .apply(lambda x: set([turn["role"] for turn in x]))
    .apply(lambda x: Counter(x)).sum()
)

def print_conv(conv):
    for turn in conv:
        print("===", turn["role"], "===")
        print(turn["content"])
# print_conv(df.sample(1)["conversations"].iloc[0])
# import pdb; pdb.set_trace()

# Save to file
if args.subsample_size is not None:
    output_filepath = os.path.join(args.output_dir, f"{args.output_filename}.n{args.subsample_size}.jsonl")
    df = df.sample(args.subsample_size, random_state=42, replace=False)
    print(f"After Subsample: {len(df)} examples")
else:
    output_filepath = os.path.join(args.output_dir, f"{args.output_filename}.jsonl")

df = df[["id", "conversations"]]
print(f"Writing {len(df)} examples to {output_filepath}")
df.to_json(output_filepath, orient="records", lines=True)
