import re
import os
import json
import openai
import logging
import pathlib
import backoff
import requests
import argparse
import traceback
import pandas as pd
import google.generativeai as genai
from tasks import get_task_iterator, Task, ToolType, ActionMode
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--action_mode", type=str, default="text_as_action")
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--task_regex_filter", type=str, default=None)
parser.add_argument("--n_tasks", type=int, default=None)
parser.add_argument("--n_turns_limit", type=int, default=10)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

def postprocess_fn(generation: str) -> str:
    generation = generation.lstrip()
    # Regex pattern to find "Answer:" or "Action:"
    pattern_answer_action = r"(Answer:|Action:)"
    matches_answer_action = list(re.finditer(pattern_answer_action, generation))

    # If no matches or only one match of Answer/Action, return the original string
    if len(matches_answer_action) <= 1:
        return generation

    # Get the index of the start of the second match of Answer/Action
    second_match_start = matches_answer_action[1].start()

    # Trim the string to end before the second match of Answer/Action
    trimmed_generation = generation[:second_match_start].strip()

    # Find the index of the end of the first "Action:"
    first_action_end_index = trimmed_generation.find("Action:") + len("Action:")
    # Check for the next occurrence of "End Action" after the first "Action:"
    end_action_index = trimmed_generation.find("End Action", first_action_end_index)
    # Determine where to trim the string
    if end_action_index != -1:
        # Trim the string to the determined index
        trimmed_generation = trimmed_generation[:end_action_index+len("End Action")].strip()

    # Check for the next occurrence of "Thought:" after the first "Action:"
    next_thought_index = trimmed_generation.find("Thought:", first_action_end_index)
    if next_thought_index != -1:
        trimmed_generation = trimmed_generation[:next_thought_index].strip()
    
    return trimmed_generation

@backoff.on_exception(
    backoff.fibo,
    # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
    (
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIConnectionError,
    ),
)
def chat_completion(messages):
    extra_kwargs = {}
    if "lemur" in args.model:
        extra_kwargs["stop"] = ["###"]

    response = openai.ChatCompletion.create(
        model=args.model,
        messages=messages,
        max_tokens=512,
        temperature=0,
        timeout=30,
        **extra_kwargs,
    )
    return response.choices[0].message["content"]

@backoff.on_exception(
    backoff.fibo,
    # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
    Exception,
    logger=logging.getLogger(),
)
def gemini_chat_completion(messages):

    ROLE_MAPPING = {
        "system": "system",
        "assistant": "model",
        "user": "user"
    }

    extra_kwargs = {}
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    extra_kwargs = {
        "safety_settings": safety_settings,
    }
    # Convert message format
    messages = [
        {
            "role": ROLE_MAPPING[message["role"]],
            "parts": [message["content"]]
        }
        for message in messages
    ]
    # merge the system message with the first user message
    # since Gemini doesn't support system message
    if messages[0]["role"] == "system":
        if len(messages) > 1:
            assert messages[1]["role"] == "user"
            # merge the system message with the first user message
            messages[1]["parts"][0] = f"{messages[0]['parts'][0]}\n\n{messages[1]['parts'][0]}"
            messages = messages[1:]
        else:
            # if there is only one message, we just set the role to user
            messages[0]["role"] = "user"

    for message in messages:
        print(message["role"], message["parts"])

    model = genai.GenerativeModel(args.model)
    response = model.generate_content(
        messages,
        **extra_kwargs
    )
    return response.text

@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
)
def claude_chat_completion(messages, **kwargs):
    # Prepend the prompt with the system message
    data = {
        "model": args.model,
        "prompt": "",
        "max_tokens_to_sample": 512,
        "temperature": 0,
    }
    for message in messages:
        if message["role"] == "user" or message["role"] == "system":
            data["prompt"] += f"\n\nHuman: {message['content']}"
        else:
            data["prompt"] += f"\n\nAssistant: {message['content']}"
    data["prompt"] += "\n\nAssistant:"

    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        pass
    else:
        raise requests.exceptions.RequestException(
            "Request failed with status code:", response.status_code
        )

    return json.loads(response.text)["completion"]


@backoff.on_exception(
    backoff.fibo,
    # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
    (
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIConnectionError,
        openai.error.APIError,
    ),
)
def davinci_completion(messages, **kwargs):

    # messages to prompt
    prompt = ''
    for message in messages:
        prompt += message['role'] + ': ' + message['content'] + '\n'
    prompt += "\nassistant:"

    query = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 512,
        "stop": [
            "\nThought:",
            "user:"
        ],
    }
    query.update(kwargs)
    response = openai.Completion.create(**query)
    resp_str = response['choices'][0]['text']
    resp_str = resp_str.strip()
    return resp_str


ACTION_MODES = [
    ActionMode.TEXT_AS_ACTION,
    ActionMode.JSON_AS_ACTION,
    ActionMode.CODE_AS_ACTION
]

def run_task(task_name: str, task: Task, action_mode: ActionMode, n_limit: int = 10):
    task: Task
    print("========================================")
    print(task_name, action_mode)
    cur_prompt = task.get_prompt(action_mode)
    # print prompt in yellow
    print("Prompt:")
    print("\033[93m" + cur_prompt + "\033[0m")
    messages = [
        {
            "role": "user",
            "content": cur_prompt
        },
    ]

    is_correct = False
    n_turns = 0
    end_reason = "limit"
    while not is_correct and n_turns < n_limit:
        try:
            if "claude" in args.model:
                generation = claude_chat_completion(messages)
            elif "davinci" in args.model:
                generation = davinci_completion(messages)
            elif "gemini" in args.model:
                generation = gemini_chat_completion(messages)
            else:
                generation = chat_completion(messages)
        except openai.error.InvalidRequestError:  # mostly due to model context window limit
            tb = traceback.format_exc()
            print(f"InvalidRequestError\n{tb}")
            end_reason = "context_limit"
            break
        except openai.error.APIError as e:  # mostly due to model context window limit
            tb = traceback.format_exc()
            print(f"APIError\n{tb}")
            assert "context" in str(e)
            end_reason = "context_limit"
            break

        print("Raw Completion:")
        print(generation)
        print("Post-processed Completion:")
        # print completion in green
        generation = postprocess_fn(generation)
        print("\033[92m" + generation + "\033[0m")
        messages.append({
            "role": "assistant",
            "content": generation
        })
        n_turns += 1

        # Parse the generation
        parsed = task.parse_generation(generation)
        content_type = parsed["type"]
        content = parsed["content"]
        if "extra_info" in parsed and parsed["extra_info"] is not None:
            extra_info = parsed["extra_info"]
        else:
            extra_info = ""

        if content_type == "action":
            execution_result = task.execute_action(content, action_mode)

            content = str(execution_result)
            if extra_info != "":
                content += "\n*Extra reminder: " + extra_info
            messages.append({
                "role": "user",
                "content": content
            })

            if task.is_single_tool_task:  # Check correctness for single tool tasks
                is_correct = task.check_answer(execution_result)
                if is_correct:
                    print("Correct for single tool task!")
                    break

        elif content_type == "answer":
            is_correct = task.check_answer(content)

            print("Expected output:", task.expected_output)
            print("Is correct:", is_correct)
            
            content = "Your answer is incorrect. Please try again. Note that you should ONLY output the answer (e.g., single number), without any other text."
            if extra_info != "":
                content += "\n*Extra reminder: " + extra_info

            if is_correct:
                break
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            messages.append({
                "role": "user",
                "content": content
            })
        # Print last message in blue
        print("\033[94m" + messages[-1]["content"] + "\033[0m")
    
    if is_correct:
        end_reason = "correct"

    return {
        "task_name": task_name,
        "is_single_tool_task": task.is_single_tool_task,
        "action_mode": action_mode.value,
        "prompt": cur_prompt,
        "messages": messages,
        "expected_output": task.expected_output,
        "is_correct": is_correct,
        "n_turns": n_turns,
        "end_reason": end_reason,
    }


def run_tools(action_mode: ActionMode, fout, outputs):
    # Filter out tasks that have been run
    finished_task_names = set([output["task_name"] for output in outputs])
    print(f"Found {len(finished_task_names)=}")

    # for task_name, task in tqdm(task_to_run):
    for i, task in enumerate(get_task_iterator()):
        task: Task
        task_name = task.name

        # Skip tasks that does not match the regex
        if args.task_regex_filter is not None \
            and re.search(args.task_regex_filter, task_name) is None:
            continue

        # Skip when max number of tasks is reached
        if args.n_tasks is not None and i >= args.n_tasks:
            break

        if task_name in finished_task_names:
            # skip tasks that have been run
            continue
        task.reset()
        cur_output = run_task(task_name, task, action_mode, args.n_turns_limit)
        fout.write(json.dumps(cur_output) + "\n")
        task.free_resource()

output_dir = f"{args.output_dir}/{args.model}"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

action_mode = args.action_mode
assert action_mode in [action_mode.value for action_mode in ACTION_MODES]
action_mode = ActionMode[action_mode.upper()]
print(f"Running {action_mode=}")

output_filepath = f"{output_dir}/{action_mode.value}.json"
outputs = []
if pathlib.Path(output_filepath).exists():
    with open(output_filepath, "r") as f:
        outputs = [json.loads(line) for line in f.readlines()]

fout = open(output_filepath, "a")
run_tools(action_mode, fout, outputs)
fout.close()
