# %%
from glob import glob
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from collections import Counter
import math

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd())) # Should be your path to the repo `mint`
# sys.path.insert(0, ROOT_DIR)
# DATA_DIR = os.path.join(ROOT_DIR, "data", "outputs")

print(f"Data directory: {DATA_DIR}")
glob_pattern = f"{DATA_DIR}/code-act-agent/**/*results.jsonl"
filepaths = list(set(glob(glob_pattern, recursive=True)))
print(f"Matching glob pattern: `{glob_pattern}`. **{len(filepaths)}** files found.")


def parse_filepath(filepath):
    # e.g., gpt-3.5-turbo-0613/F=gpt-3.5-turbo-16k-0613/PHF=no_GT-textual/max5_p2+tool+cd/code_generation/humaneval/results.jsonl
    # e.g., gpt-3.5-turbo-0613/F=None/max5_p2+tool+cd/code_generation/humaneval/results.jsonl
    splited = filepath.replace(DATA_DIR, "").lstrip("/").split("/")
    
    agent_model_name = splited[0]
    feedback_model_name = splited[1].split("=")[1]
    if feedback_model_name != "None":
        feedback_setting = splited[2]
    else:
        feedback_setting = "None"
    task_name = splited[-2]
    task_type = splited[-3]
    exp_setting = splited[-4]
    return {
        "agent_model_name": agent_model_name,
        "feedback_model_name": feedback_model_name,
        "feedback_setting": feedback_setting,
        "task_name": task_name,
        "task_type": task_type,
        "exp_setting": exp_setting,
        "filepath": filepath,
    }

df = pd.DataFrame(list(map(parse_filepath, filepaths)))

def load_results(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception as e:
                print(f"Error loading {filepath}: {e}\n{line}")
                globals()["error_line"] = line
    return pd.DataFrame(results)

df["results"] = df.filepath.apply(load_results)

all_results = []
for row in df.itertuples():
    row.results["agent_model_name"] = row.agent_model_name
    row.results["feedback_model_name"] = row.feedback_model_name
    row.results["feedback_setting"] = row.feedback_setting
    row.results["exp_setting"] = row.exp_setting
    row.results["task_name"] = row.task_name
    row.results["task_type"] = row.task_type
    all_results.append(row.results)


all_results = pd.concat(all_results)
def get_stats(row):
    state = row["state"]
    task = row["task"]
    return {
        "task_id": task["task_id"],
        "n_turns": len(state["history"]) // 2,
        "success": state["success"],
        "agent_action_count": state["agent_action_count"],
        "token_counter": {'a': Counter(state["token_counter"]), 'b': 1},
        "terminate_reason": state["terminate_reason"],
    }


# combine this with the original dataset
stats = all_results.apply(get_stats, axis=1, result_type="expand")
all_results = pd.concat([all_results, stats], axis=1)

# turn bool to int
all_results['success'] = all_results['success'].astype(int)

all_results_unfiltered = all_results.copy()


# %%
# Remove duplicates in case of weird bugs
all_results_no_dup = all_results.drop_duplicates(
    subset=["task_type", "task_name", "task_id", "agent_model_name", "feedback_model_name", "feedback_setting", "exp_setting"],
    keep="first"
)
if len(all_results_no_dup) != len(all_results):
    print(f"WARNING: Removed {len(all_results) - len(all_results_no_dup)} duplicated rows.")
    all_results = all_results_no_dup

# Sanity check of experiments - check whether they are all completed
all_results_count = all_results.groupby([
    "agent_model_name",
    "feedback_model_name",
    "feedback_setting",
    "exp_setting",
    "task_type",
    "task_name",
])["task_id"]\
.count().unstack().fillna(0)\
.sum(axis=1).unstack().fillna(0).astype(int)

print("====== Experiment Results Count ======")
print(all_results_count)


# %%
# Filter out experiments that are not completed

# find all index that are not [136, 134, 320]
GLOBAL_MAX = all_results_count.max()
assert (GLOBAL_MAX == pd.Series([136, 134, 316], index=["code_generation", "decision_making", "reasoning"])).all()
def _exp_completed(row):
    return (row == GLOBAL_MAX).all()

completed_exp = all_results_count.apply(_exp_completed, axis=1)
# select only completed exp
completed_exp = completed_exp.drop(completed_exp[completed_exp == False].index)#.reset_index().drop(columns=[0])

completed_exp_lst = set(map(tuple, completed_exp.reset_index().drop(columns=[0]).to_numpy().tolist()))
# agent_model_name	feedback_model_name	feedback_setting	exp_setting
# completed_exp_lst
_completed_mask = all_results.apply(lambda row: (row["agent_model_name"], row["feedback_model_name"], row["feedback_setting"], row["exp_setting"]) in completed_exp_lst, axis=1)
print(f"Before filtering: {len(all_results)}")
not_completed = all_results[~_completed_mask]
all_results = all_results[_completed_mask]
print(f"After filtering: {len(all_results)}")


# %%
_n = not_completed.groupby(["agent_model_name", "feedback_model_name", "feedback_setting", "exp_setting", "task_type"])["task_id"].count().reset_index()[["agent_model_name", "feedback_model_name", "feedback_setting", "exp_setting", "task_type"]]

# only print agent_model_name & feedback_model_name that are not completed
print(
    "Incomplete experiments (agent_model_name, feedback_model_name, exp_setting):\n",
    _n[["agent_model_name", "feedback_model_name", "exp_setting"]].drop_duplicates()
)


# %% [markdown]
# # Results

# %%
def generate_table(
    query, mode = 'sr', return_raw = False,
    consider_exceed_context = False,
    by_task_name = False,
):
    results = all_results.query(query)
    n_invalid_actions = results['agent_action_count'].apply(lambda x: x['invalid_action'])
    n_turns = results['n_turns']
    n_error = results['agent_action_count'].apply(lambda x: x.get('error', 0))

    results = results.assign(
        n_invalid_actions=n_invalid_actions,
        n_turns=n_turns,
        n_error=n_error,
    )
    if consider_exceed_context:
        results["final_length_exceeds_4096"] = results["final_length_exceeds_4096"].fillna(False)
        results["success"] = results.apply(lambda row: int(row["success"] and not row["final_length_exceeds_4096"]), axis=1)

    results_raw = results.copy()
    # draw a table with the performance on four tasks as the horizontal axis and the micro mean success rate and different models as the vertical axis. display it with seaborn
    # firstly calculate micro mean
    if mode == 'sr':
        micro_sr = results.groupby("agent_model_name")['success'].mean()
        if by_task_name:
            results_sr = results.groupby(["agent_model_name", "task_name"])['success'].mean().unstack()
        else:
            results_sr = results.groupby(["agent_model_name", "task_type"])['success'].mean().unstack()
        results_sr['avg_micro'] = micro_sr
        results = (results_sr * 100).round(2)
    elif mode == "invalid_count":
        micro_invalid_count = results.groupby("agent_model_name")['n_invalid_actions'].mean()
        results_invalid_count = results.groupby(["agent_model_name", "task_type"])['n_invalid_actions'].mean().unstack()
        results_invalid_count['avg_micro'] = micro_invalid_count

        micro_n_turns = results.groupby("agent_model_name")['n_turns'].mean()
        results_n_turns = results.groupby(["agent_model_name", "task_type"])['n_turns'].mean().unstack()
        results_n_turns['avg_micro'] = micro_n_turns

        micro_n_error = results.groupby("agent_model_name")['n_error'].mean()
        results_n_error = results.groupby(["agent_model_name", "task_type"])['n_error'].mean().unstack()
        results_n_error['avg_micro'] = micro_n_error

        results = pd.concat(
            [results_invalid_count, results_n_turns, results_n_error],
            axis=1,
            keys=["invalid_count", "n_turns", "n_error"]
        ).round(2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if return_raw:
        return results, results_raw
    else:
        return results

# %% [markdown]
# ## Tool-augmented Task-solving (SR vs. k)

# %%

# Combine results
def get_interaction_turns(**kwargs):
    interaction_turns = []
    for setting in ['max1_p1+tool+cd', 'max2_p2+tool+cd', 'max3_p2+tool+cd', 'max4_p2+tool+cd', 'max5_p2+tool+cd']:
        _cur_table, _cur_table_raw = generate_table(
            f"exp_setting == '{setting}' and feedback_setting == 'None' and feedback_model_name == 'None'",
            mode="sr",
            return_raw=True,
            **kwargs
        )
        _cur_table = _cur_table.unstack().to_frame().rename(columns={0: 'success_rate'}).reset_index()
        cur_turn = int(re.search(r'max(\d+)_', setting).group(1))
        _cur_table['n_turns'] = cur_turn
        interaction_turns.append(_cur_table)
        _turn_vs_success = _cur_table_raw[["agent_model_name", "success"]]
        _turn_vs_success = _turn_vs_success.assign(n_turns=cur_turn)

    interaction_turns = pd.concat(interaction_turns)
    return interaction_turns

interaction_turns = get_interaction_turns()
interaction_turns_by_task_name = get_interaction_turns(by_task_name=True)
# Run regression and generate a table
assert interaction_turns["agent_model_name"].nunique() == 1, f"Only one agent model is allowed, but got {interaction_turns['agent_model_name'].nunique()}: {interaction_turns['agent_model_name'].unique()}"

def run_regression_on_row(series):
    if not len(series.dropna()):
        print(series)
        return {}
    x = list(series.index)
    y = series.values
    _df = pd.DataFrame({"x": x, "y": y})
    model = sm.ols(formula='y ~ x', data=_df)
    results = model.fit()
    return {
        **{k: v for k, v in results.params.to_dict().items()},
        "rsquared": results.rsquared,
        # "pvalues": results.pvalues.to_dict()
        **{f"pvalue_{k}": v for k, v in results.pvalues.to_dict().items()}
    }

def get_sr_vs_k_data(interaction_turns, index_key="task_type"):
    _view = interaction_turns.set_index([index_key, "n_turns"]).drop(columns=["agent_model_name"]).unstack()["success_rate"]
    _view = pd.concat([_view, _view.apply(run_regression_on_row, axis=1, result_type='expand')], axis=1)
    _view.sort_index(inplace=True)
    return _view
print("====== SR vs. k ======")
_view = get_sr_vs_k_data(interaction_turns)
_view_by_task_name = get_sr_vs_k_data(interaction_turns_by_task_name, index_key="task_name")
print(_view)
sr_vs_k_table_output_path = os.path.join(OUTPUT_DIR, "sr_vs_k.json")
_view.to_json(
    sr_vs_k_table_output_path,
    orient="index", indent=2
)
print(f"SR vs. k table saved to {sr_vs_k_table_output_path}")
print(_view_by_task_name)
sr_vs_k_table_by_task_name_output_path = os.path.join(OUTPUT_DIR, "sr_vs_k_by_task_name.json")
_view_by_task_name.to_json(
    sr_vs_k_table_by_task_name_output_path,
    orient="index", indent=2
)
print(f"SR vs. k table saved to {sr_vs_k_table_by_task_name_output_path}")

# %% [markdown]
# ## Ability to Leverage Natural Language Feedback (Delta Feedback)

# %%
main_table_sr = generate_table(
    "exp_setting == 'max5_p2+tool+cd' and feedback_setting == 'None' and feedback_model_name == 'None'", mode="sr"
)
print("====== SR @ 5 ======")
if len(main_table_sr) == 0:
    print("No SR @ 5 results found.")
    exit(0)

print(main_table_sr)

# generate feedback ablation results (gpt-4-0613)
gpt4_feedback = generate_table(
    "exp_setting == 'max5_p2+tool+cd' and feedback_setting == 'PHF=no_GT-textual' and feedback_model_name == 'gpt-4-0613'",
    mode="sr",
)
if len(gpt4_feedback) == 0:
    print("No GPT-4 feedback results found.")
    exit(0)

gpt4_feedback_diff = gpt4_feedback - main_table_sr

combined_table = pd.concat([
    main_table_sr, gpt4_feedback, gpt4_feedback_diff
    ], axis=1, keys=["SR", "SR (gpt-4-0613)", "Diff"]
).swaplevel(axis=1).sort_index(axis=1)
print("====== SR vs. SR (gpt-4-0613) ======")
print(combined_table)

feedback_table_output_path = os.path.join(OUTPUT_DIR, "sr_w_feedback.json")
combined_table.to_json(
    feedback_table_output_path,
    orient="index", indent=2
)
print(f"SR w/ feedback table saved to {feedback_table_output_path}")
