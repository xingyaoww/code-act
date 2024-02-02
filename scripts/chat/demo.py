import re
import json
import openai
import pathlib
import requests
import argparse
from termcolor import colored
from transformers import AutoTokenizer
from typing import List, Dict
from datetime import datetime

if openai.__version__ >= "0.28.1":
    raise RuntimeError(
        "Please use the compatbile version of openai (<=0.28.1) to use this script."
    )

class ClientJupyterKernel:
    def __init__(self, url, conv_id):
        self.url = url
        self.conv_id = conv_id
        print(f"ClientJupyterKernel initialized with url={url} and conv_id={conv_id}")

    def execute(self, code):
        payload = {"convid": self.conv_id, "code": code}
        response = requests.post(self.url, data=json.dumps(payload))
        response_data = response.json()
        if response_data["new_kernel_created"]:
            print(f"New kernel created for conversation {self.conv_id}")
        return response_data["result"]


class Generator:
    def __init__(self, model_name: str, openai_base_url: str):
        self.model_name = model_name
        self.openai_base_url = openai_base_url
        print(
            f"Generator initialized with model_name={model_name} and openai_base_url={openai_base_url}"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        do_sample: bool = True,
        max_new_tokens: int = 512,
        stop_sequences: List[str] = ["<|im_end|>"],
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if do_sample else 0.0,
            max_tokens=max_new_tokens,
            top_p=top_p if do_sample else 1.0,
            stop=stop_sequences,
        )
        return completion.choices[0].message.content


SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
The assistant can interact with an interactive Python (Jupyter Notebook) environment and receive the corresponding output when needed. The code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
The assistant should attempt fewer things at a time instead of putting too much code in one <execute> block. The assistant can install packages through PIP by <execute> !pip install [package needed] </execute> and should always import packages and define variables before starting to use them.
The assistant should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
The assistant's response should be concise, but do express their thoughts.
"""

# NOTE: You may also include the following information in the system message if you have pre-defined tools for the assistant to execute.
# Tool function available (already imported in <execute> environment):
# [1] google_search(query: str, num_results: int = 1) -> dict
# Search google for the given query. You should rely on this to get most up-to-date information. Do not make things up.
# For example: \"<execute>\"google_search(\"Hello world\") \"</execute>\"
# [2] get_url_content(url: str) -> str
# Get content from URL. You can use this when you want to access more information from an URL.
# [3] get_url_html(url: str) -> str
# Get HTML from URL (could be messy).



class Agent:
    COLOR_MAP = {
        "user": "green",
        "execution_output": "yellow",
        "assistant": "blue",
        "system": "red",
    }

    def __init__(
        self,
        generator: Generator,
        code_executor: ClientJupyterKernel,
        system_message: str = SYSTEM_MESSAGE,
        conv_id: str = None,
        **kwargs,
    ):
        self.messages = [
            {"role": "system", "content": system_message},
        ]
        self.kwargs = {
            "stop_sequences": ["<|im_end|>", "</execute>"],
            "do_sample": False,
            "max_new_tokens": 512,
            **kwargs,
        }
        self.generator = generator
        self.code_executor = code_executor
        self.conv_id = conv_id
        # print the messages
        for message in self.messages:
            self.print_message(message)

    def print_message(self, message):
        # bold print the role
        print("-" * 20)
        print(
            colored(
                message["role"].upper(), self.COLOR_MAP[message["role"]], attrs=["bold"]
            )
        )
        print(colored(message["content"], self.COLOR_MAP[message["role"]]))

    def handle_execution(self, completion: str, code_executor):
        # use regex to capture the code
        code = re.search(r"<execute>(.*)</execute>", completion, re.DOTALL)
        # check if the code is valid
        if code is not None:
            code = code.group(1)
            # execute the code
            result = code_executor.execute(code)
            # return the result
            return result
        return None

    def handle_user_message(self, message, n_max_executions=3):
        # append the message
        self.messages.append({"role": "user", "content": message})
        self.print_message(self.messages[-1])

        execution_count = 0
        while (
            self.messages[-1]["role"] == "user" and execution_count < n_max_executions
        ):
            response = self.generator.generate(self.messages, **self.kwargs)
            self.messages.append({"role": "assistant", "content": response})
            self.print_message(self.messages[-1])

            execution_output = self.handle_execution(response, self.code_executor)
            if execution_output is not None:
                execution_count += 1
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Execution Output:\n" + execution_output,
                    }
                )
                self.print_message(
                    {"role": "execution_output", "content": execution_output}
                )

        if execution_count == n_max_executions:
            assert self.messages[-1]["role"] == "user"
            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"I have reached the maximum number of executions ({n_max_executions=}). Can you assist me or ask me another question?",
                }
            )
            self.print_message(self.messages[-1])

    def run(self):
        while True:
            message = input("User Input> ")
            if message == "exit":
                self.save()
                break
            self.handle_user_message(message)

    def save(self):
        pathlib.Path("conv_data").mkdir(exist_ok=True)
        path = f"conv_data/{self.conv_id}.json"
        with open(path, "w") as f:
            json.dump(self.messages, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, default="xingyaoww/CodeActAgent-Mistral-7b-v0.1")
parser.add_argument("--openai_base_url", type=str, required=True, default="http://localhost:8080/v1")
parser.add_argument("--jupyter_kernel_url", type=str, required=True, default="http://localhost:8081/execute")
args = parser.parse_args()

CONV_ID = "demo-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

code_executor = ClientJupyterKernel(args.jupyter_kernel_url, CONV_ID)
generator = Generator(args.model_name, args.openai_base_url)
agent = Agent(generator, code_executor, conv_id=CONV_ID)
agent.run()
