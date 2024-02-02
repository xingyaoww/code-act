import os
import ast
import json
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_dir', type=str, help='Path to the checkpoint directory')
parser.add_argument('--output_filepath', type=str, default=None, help='Path to the output file')
args = parser.parse_args()


eval_dir = os.path.join(args.ckpt_dir, 'eval')
assert os.path.exists(eval_dir), f'Eval dir {eval_dir} does not exist'

eval_tasks = os.listdir(eval_dir)
eval_tasks = [
    task
    for task in eval_tasks
    if os.path.isdir(os.path.join(eval_dir, task)) and task != "logs"
]

print(f"Found {len(eval_tasks)} eval tasks: {' '.join(eval_tasks)}")

def _read_json(filepath):
    assert os.path.exists(filepath), f'file {filepath} does not exist'
    with open(filepath, 'r') as f:
        return json.load(f)

def handle_mmlu(task_dir):
    results = _read_json(os.path.join(task_dir, 'results.json'))
    return {
        "accuracy": results["overall_acc"]
    }

def handle_gsm8k(task_dir):
    results = _read_json(os.path.join(task_dir, 'gsm8k_res.jsonl.metrics.json'))
    return {
        "accuracy": results["acc"]
    }

def handle_math(task_dir):
    results = _read_json(os.path.join(task_dir, 'results.json'))
    return {
        "accuracy": results["overall_accuracy"]["accuracy"]
    }

def handle_mtbench(task_dir):
    mt_bench_score = 0.0
    with open(os.path.join(task_dir, 'mt-bench-result.txt')) as f:
        for line in f:
            if line.startswith("mt-bench") \
                and "mt-bench 1" not in line and "mt-bench 2" not in line:
                mt_bench_score = float(
                    line.replace("mt-bench", "").strip()
                )
    return {
        "mt-bench": mt_bench_score
    }

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
        "k=1": results["avg_micro"]["1"],
        "k=2": results["avg_micro"]["2"],
        "k=3": results["avg_micro"]["3"],
        "k=4": results["avg_micro"]["4"],
        "k=5": results["avg_micro"]["5"],
        "Slope": results["avg_micro"]["x"],
    }
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
    


def handle_humaneval(task_dir):
    result_filepath = os.path.join(task_dir, 'result.txt')
    # find the line that has 'pass@1'
    with open(result_filepath, 'r') as f:
        for line in f:
            if 'pass@1' in line:
                break
    # parse the line
    results = ast.literal_eval(line)
    assert len(results) == 3
    assert 'pass@1' in results and 'pass@10' in results and 'pass@100' in results
    return results


def handle_science_world(task_dir):
    results = _read_json(os.path.join(task_dir, 'results.json'))
    return {
        "score": results["average"]
    }

def handle_miniwob_plus_plus(task_dir):
    results = _read_json(os.path.join(task_dir, 'results.json'))
    return {
        "success_rate": results["average"]
    }

def handle_bbh(task_dir):
    bbh_results = _read_json(os.path.join(task_dir, 'bbh.json'))
    bbh_cot_results = _read_json(os.path.join(task_dir, 'bbh_cot.json'))
    return {
        "exact_match": bbh_results["exact_match"],
        "exact_match_by_task": bbh_results["exact_match_by_task"],
        
        "exact_match_cot": bbh_cot_results["exact_match"],
        "exact_match_by_task_cot": bbh_cot_results["exact_match_by_task"],
    }


TASK_TO_HANDLERS = {
    'mmlu': handle_mmlu,
    'mmlu_chat': handle_mmlu,
    'gsm8k': handle_gsm8k,
    'math': handle_math,
    'human_eval': handle_humaneval,
    'human_eval_chat': handle_humaneval,
    'mint-bench': handle_mint_bench,
    "science-world": handle_science_world,
    "miniwob++": handle_miniwob_plus_plus,
    "bbh": handle_bbh,
    "mt-bench": handle_mtbench,
}
agg_results = {}

for task in eval_tasks:
    task_dir = os.path.join(eval_dir, task)
    print(f'Handling task {task} in {task_dir}')
    
    # check if task is DONE
    if not os.path.exists(os.path.join(task_dir, 'DONE')):
        print(f'Task {task} is not DONE yet, skipping')
        continue

    if task not in TASK_TO_HANDLERS:
        print(f'WARNING: No handler for task {task}, skipping')
        continue

    handler = TASK_TO_HANDLERS[task]
    result = handler(task_dir)
    print(f"- {task}: {result}")
    for metric, value in result.items():
        agg_results[f'{task}/{metric}'] = value

print(f"Aggregated results: {agg_results}")
with open(os.path.join(eval_dir, 'agg_results.json'), 'w') as f:
    json.dump(agg_results, f, indent=4)

# data/ckpts/Llama-2-7b-t4-p1-oct28_full6728+sharegpt4_6599_pack4096-lr1e-5-warmup50-bs32-seq4096-ep5/hf/mint_agent_iter_1185
# TO data/results/Llama-2-7b-t4-p1-oct28_full6728+sharegpt4_6599_pack4096-lr1e-5-warmup50-bs32-seq4096-ep5/hf/mint_agent_iter_1185.json
result_filepath = os.path.join("data", "results", args.ckpt_dir.lstrip('data/ckpts')).rstrip('/') + '.json'
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
