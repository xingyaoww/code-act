import json
import re

with open('prompt_orig.json') as f:
    examples: dict[str, str] = json.load(f)

processed = {}
for task_num, example in examples.items():
    conversation = []
    pattern = r'(.+?)\n> ([^\n]+)\n'
    for match in re.finditer(pattern, example, re.DOTALL):
        observation = match[1]
        choice = match[2]
        if choice.startswith('think:'):
            choice = choice.replace('think:', 'Think:', 1)
        else:
            choice = 'Action: ' + choice
        conversation += [observation, choice]
    processed[task_num] = conversation
    assert len(conversation) % 2 == 0

with open('prompt.json', 'w') as f:
    json.dump(processed, f, indent=4)