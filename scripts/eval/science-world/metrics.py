from pathlib import Path
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="logs")
args = parser.parse_args()

scores = []
for i in range(30):
    file = Path(args.input_dir) / f'task{i}-score.txt'
    try:
        s = file.open().read()
    except:
        print(f'Warning: {file} not found')
        continue
    score = re.search(r'Average score: ([0-9\.]*)', s)[1]
    x = float(score)
    scores.append(x)

with open(Path(args.input_dir) / 'results.json', 'w') as f:
    json.dump({
        "scores": scores,
        "average": sum(scores) / len(scores),
    }, f)
