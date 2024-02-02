import os
import ast
import json
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('eval_dir', type=str, help='Path to the checkpoint directory')
parser.add_argument('--output_filepath', type=str, default=None, help='Path to the output file')
args = parser.parse_args()

def _read_json(filepath):
    assert os.path.exists(filepath), f'file {filepath} does not exist'
    with open(filepath, 'r') as f:
        return json.load(f)

def handle_mint_bench(task_dir):
    TASK_COUNT = {
        "humaneval": 45,
        "mbpp": 91,
        "alfworld": 134,
        "gsm8k": 48,
        "hotpotqa": 43,
        "math": 100,
        "mmlu": 76,
        "theoremqa": 49,
    }
    TASK_MAPPING = {
        # in-domain reasoning
        "alfworld": "decision_making-id",
        "math": "reasoning-id",
        "hotpotqa": "reasoning-id",
        
        # out-of-domain reasoning
        "gsm8k": "reasoning-ood",
        "mmlu": "reasoning-ood",
        "theoremqa": "reasoning-ood",
        "humaneval": "codegen-ood",
        "mbpp": "codegen-ood",
    }

    results = _read_json(os.path.join(task_dir, 'results', 'sr_vs_k.json'))
    results_by_task_name = _read_json(os.path.join(task_dir, 'results', 'sr_vs_k_by_task_name.json'))
    assert "avg_micro" in results
    # TODO: implement w/ feedback
    ret = {
        "k=5": results["avg_micro"]["5"],
        "Slope": results["avg_micro"]["x"],
    }
    for i in range(1, 5):
        if f"{i}" in results["avg_micro"]:
            ret[f"k={i}"] = results["avg_micro"][f"{i}"]

    # re-calculate the micro-average using task_based
    micro_avg = {}  # map task_type to (sum, count)
    for task_name, task_results in results_by_task_name.items():
        if task_name not in TASK_MAPPING:
            continue
        task_type = TASK_MAPPING[task_name]
        if task_type not in micro_avg:
            micro_avg[task_type] = [0.0, 0]
        micro_avg[task_type][0] += task_results["5"] * TASK_COUNT[task_name]
        micro_avg[task_type][1] += TASK_COUNT[task_name]
    for task_type, (sum, count) in micro_avg.items():
        ret[f"micro-{task_type}-k=5"] = sum / count
    return ret


TASK_TO_HANDLERS = {
    # 'mmlu': handle_mmlu,
    # 'mmlu_chat': handle_mmlu,
    # 'gsm8k': handle_gsm8k,
    # 'math': handle_math,
    # 'human_eval': handle_humaneval,
    # 'human_eval_chat': handle_humaneval,
    'mint-bench': handle_mint_bench,
    # "science-world": handle_science_world,
    # "miniwob++": handle_miniwob_plus_plus,
    # "bbh": handle_bbh,
    # "mt-bench": handle_mtbench,
}
agg_results = {}

# ONLY DO MINT
task = 'mint-bench'
task_dir = args.eval_dir
print(f'Handling task MINT in {task_dir}')

handler = TASK_TO_HANDLERS[task]
result = handler(task_dir)
print(f"- {task}: {result}")
for metric, value in result.items():
    agg_results[f'{task}/{metric}'] = value

print(f"Aggregated results: {agg_results}")
with open(os.path.join(task_dir, 'agg_results.json'), 'w') as f:
    json.dump(agg_results, f, indent=4)

result_filepath = os.path.join("data", "results", args.eval_dir.lstrip('data/models/mint-bench')).rstrip('/') + '.json'
pathlib.Path(os.path.dirname(result_filepath)).mkdir(parents=True, exist_ok=True)
with open(result_filepath, 'w') as f:
    json.dump(agg_results, f, indent=4)
print(f'Saved aggregated results to {result_filepath}')

if args.output_filepath is not None:
    # Save an additional copy of the aggregated results
    output_filepath = args.output_filepath
    # create folder if not exists
    pathlib.Path(os.path.dirname(output_filepath)).mkdir(parents=True, exist_ok=True)

    print(f'Saving aggregated results to {output_filepath}')
    with open(output_filepath, 'w') as f:
        json.dump(agg_results, f, indent=4)
