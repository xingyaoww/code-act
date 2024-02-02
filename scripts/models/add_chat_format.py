import os
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='Path to the model')
args = parser.parse_args()

print(f"Loading tokenizer from {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# ChatML Template
# https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Back-up the original tokenizer config
os.rename(
    os.path.join(args.model_path, 'tokenizer_config.json'),
    os.path.join(args.model_path, 'tokenizer_config.json.bak')
)

# save the tokenizer config
tokenizer.save_pretrained(args.model_path)
