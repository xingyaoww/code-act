import os
import time
import openai
import argparse
import traceback
from tqdm import tqdm
from typing import List
from human_eval.data import write_jsonl, read_problems

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=200)
# for pass@1
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/docs/README.md?plain=1#L44
parser.add_argument("--temperature", type=float, default=0.2)
args = parser.parse_args()

problems = read_problems()

# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/bigcode_eval/tasks/humaneval.py#L54C13-L54C87
STOP_WORDS =["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]

def generate_completions(prompt, num_samples_per_task) -> List[str]:
    while True:
        try:
            response = openai.Completion.create(
                model=args.model,
                prompt=prompt,
                max_tokens=512,
                temperature=args.temperature,
                n=num_samples_per_task,
                stop=STOP_WORDS,
                api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_KEY"),
            )
            # return response["choices"][0]["text"]
            return [
                choice["text"]
                for choice in response["choices"]
            ]
        except Exception as e:
            traceback.print_exc()
            time.sleep(1)
            continue

num_samples_per_task = args.num_samples_per_task
pbar = tqdm(total=len(problems) * num_samples_per_task)
samples = []
for task_id in problems:
    samples.extend([
        dict(task_id=task_id, completion=completion)
        for completion in generate_completions(problems[task_id]["prompt"], num_samples_per_task)
    ])
    pbar.update(num_samples_per_task)
pbar.close()
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)
