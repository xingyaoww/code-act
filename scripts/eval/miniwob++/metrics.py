import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="history")
args = parser.parse_args()

correct = 0
total = 0

task_dir = Path(args.input_dir) / 'outputs'
for task in task_dir.iterdir():
    if not task.is_dir():
        continue
    task_orig = task
    task = next(next(task.iterdir()).iterdir())
    episode_count = 0
    for episode in task.iterdir():
        if not episode.suffix == '.txt':
            continue
        is_correct = 'success' in episode.name
        is_fail = 'fail'
        correct += is_correct
        total += 1
        episode_count += 1

with open(Path(args.input_dir) / 'results.json', 'w') as f:
    json.dump({
        "correct": correct,
        "total": total,
        "average": correct / total,
    }, f)

print(f'\t{correct:3} / {total:3} = {(correct / total):.4}')
