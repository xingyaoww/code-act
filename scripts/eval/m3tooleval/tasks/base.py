import re
import ast
import copy
import traceback
from tqdm import tqdm
from typing import Optional, Mapping, Any, Iterable, List, Tuple, Union, Callable
from collections import namedtuple
from enum import Enum

from .repl import PythonREPL

ToolType = namedtuple("ToolType", [
    "name",
    "description",
    "function",
    "fn_signature",
])

class ActionMode(Enum):
    TEXT_AS_ACTION = "text_as_action"
    JSON_AS_ACTION = "json_as_action"
    CODE_AS_ACTION = "code_as_action"

class Task:
    def __init__(
        self,
        name: str,
        tools: Union[Mapping[str, ToolType], Callable],
        instruction: str,
        expected_output: Any,
        is_single_tool_task: bool = True
    ):
        self.name = name
        self.tools = tools
        self.instruction = instruction
        self.expected_output = expected_output
        self.is_single_tool_task = is_single_tool_task
        self.print_task()
        self.reset()

    def reset(self) -> None:
        # if tools is a function, call it to get the tools
        if callable(self.tools):
            self.tools = self.tools()
        assert isinstance(self.tools, Mapping)
        self._ns = {tool_name: tool.function for tool_name, tool in self.tools.items()}
        self.repl = None

    def print_task(self) -> None:
        print('=' * 30)
        print(f"Task: {self.name}")
        print('-' * 30)
        print(f"Instruction: {self.instruction}")
        print('-' * 30)
        print(f"Expected Output: {self.expected_output}")

    def get_prompt(self, action_mode: ActionMode) -> str:
        tool_desc = "You have access to the following tools:\n"
        for i, (tool_name, tool) in enumerate(self.tools.items()):
            tool_desc += f"[{i+1}] {tool_name}: {tool.description}\n"
            tool_desc += f"    Signature: {tool.fn_signature}\n"

        res = tool_desc + "\n"
        if action_mode == ActionMode.TEXT_AS_ACTION:
            res += "You can use the tools by outputing the tool name followed by its arguments, delimited by commas.\n"
            res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
            res += "Example: 'Action: tool_name, argument_1 End Action'\n"
            res += "You can only invoke one tool at a time.\n"
        elif action_mode == ActionMode.JSON_AS_ACTION:
            res += "You can use the tools by outputing a JSON object with the following fields:\n"
            res += "  - 'tool': the name of the tool\n"
            res += "  - 'args': a list of arguments to the tool\n"
            res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
            res += "Example: 'Action: {\"tool\": \"tool_name\", \"args\": [\"argument_1\"]} End Action'\n"
            res += "You can only invoke one tool at a time.\n"
        elif action_mode == ActionMode.CODE_AS_ACTION:
            res += "You can use the tools by outputing a block of Python code that invoke the tools.\n"
            res += "You may use for-loops, if-statements, and other Python constructs when necessary.\n"
            res += "Be sure to print the final answer at the end of your code.\n"
            res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
            res += "Example: 'Action:\ntool_name(argument_1)\nEnd Action'\n"

        res = res + "\nNow, let's get started!\n\n"
        res = res + f"Instruction: {self.instruction}"
        res = res + "\nYou can optionally express your thoughts using natural language before your action. For example, 'Thought: I want to use tool_name to do something. Action: <your action to call tool_name> End Action'."
        res = res + "\nNote that your output should always contain either 'Action:' or 'Answer:', but not both."
        res = res + "\nWhen you are done, output the result using 'Answer: your answer'"
        res = res + "\nPlease ONLY output the answer (e.g., single number), without any other text."
        return res

    def parse_generation(self, generation: str) -> Optional[Mapping[str, Any]]:
        if "Answer:" in generation and "Action:" in generation:
            return {
                "type": "invalid",
                "content": "Invalid generation. Your output should contain either 'Action:' or 'Answer:', but not both."
            }

        if "Answer:" in generation:
            # find the first answer
            if generation.count("Answer:") > 1:
                # get the first answer
                answer = generation.split("Answer:")[1].strip()
                extra_info = "You have output more than one answer. Only the first answer will be used."
            else:
                answer = generation[generation.find("Answer:") + len("Answer:"):].strip()
                extra_info = None
            return {
                "type": "answer",
                "content": answer,
                "extra_info": extra_info
            }
        elif "Action:" in generation:
            if generation.count("Action:") > 1:
                action = generation.split("Action:")[1].lstrip()
                extra_info = "You have output more than one action. Only the first action will be used."
            else:
                action = generation[generation.find("Action:") + len("Action:"):].lstrip()
                extra_info = None

            if "End Action" in action: # remove the "End Action" part
                action = action[:action.find("End Action")]
            return {
                "type": "action",
                "content": action,
                "extra_info": extra_info
            }
        else:
            return {
                "type": "invalid",
                "content": "Invalid generation. Your output should contain either 'Action:' or 'Answer:'"
            }

    def check_answer(self, answer: str) -> bool:
        # Check if the answer is correct
        try:
            # Directly compare the answer
            if answer == self.expected_output or \
            (self._try_to_convert_to_correct_type(answer)
             == self._try_to_convert_to_correct_type(self.expected_output)):
                return True
            
            answer = ast.literal_eval(answer)
            if answer == self.expected_output \
                (isinstance(self.expected_output, list) and Task.compare_list(answer, self.expected_output)) \
                or (self._try_to_convert_to_correct_type(answer)
                == self._try_to_convert_to_correct_type(self.expected_output)):
                return True
        except:
            pass

        if str(answer) == str(self.expected_output):
            return True
        return False

    @staticmethod
    def compare_list(a: List[Any], b: List[Any]) -> bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i] and \
                (Task._try_to_convert_to_correct_type(a[i])
                != Task._try_to_convert_to_correct_type(b[i])):
                return False
        return True

    @staticmethod
    def _try_to_convert_to_correct_type(s: str) -> Any:
        # try int, then float, then str
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    def execute_action(self, action: str, action_mode: ActionMode) -> Optional[str]:

        if action_mode == ActionMode.CODE_AS_ACTION:
            if not self.repl:
                self.repl = PythonREPL(self._ns)

            # directly execute the code
            obs = self.repl(action)
            # Extract the observation ONLY (remove initial 'Out[0]: ')
            obs = re.sub(r"Out\[\d+\]:", "", obs)
            try:
                return ast.literal_eval(obs.strip())
            except Exception as e:
                return obs

        if action_mode == ActionMode.TEXT_AS_ACTION:
            try:
                tool_name, *args = action.split(",")
                tool_name = tool_name.strip()
                args = [self._try_to_convert_to_correct_type(arg.strip()) for arg in args]
            except Exception as e:
                traceback.print_exc()
                return f"Invalid action. You should output the tool name followed by its arguments, delimited by commas. Error: {e}"
        elif action_mode == ActionMode.JSON_AS_ACTION:
            import json
            try:
                action = json.loads(action)
                tool_name = action["tool"]
                args = action["args"]
            except Exception as e:
                traceback.print_exc()
                return f"Invalid action. You should output a JSON object with the following fields: 'tool' (the name of the tool), 'args' (a list of arguments to the tool). Error: {e}"
        if not isinstance(args, list):
            args = [args]
        return self.execute_non_code(tool_name, *args)

    def execute_non_code(self, tool_name: str, *args, **kwargs) -> str:
        if tool_name not in self.tools:
            return f"Cound not find tool with name {tool_name}"

        tool = self.tools[tool_name]
        try:
            res = tool.function(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return f"Failed to execute tool {tool_name} with args {args}. Did you try to invoke more than one tool at a time?"
        res = str(res)
        # Truncate the result
        if len(res) > 2000:
            res = res[:2000] + "...\n[Output Truncated]"
        print(f"{tool_name}(*args={args}, **kwargs={kwargs}) -> {res})")
        return res

    def free_resource(self) -> None:
        if self.repl:
            del self.repl

task_iterators: List[Tuple[Iterable[Task], int]] = []

def register_task_iterator(task_iter: Iterable[Task], length: int) -> None:
    task_iterators.append((task_iter, length))

def get_task_iterator() -> Iterable[Task]:
    generators, lengths = zip(*task_iterators)
    pbar = tqdm(total=sum(lengths))
    for generator, length in zip(generators, lengths):
        for task in generator:
            yield task
            pbar.update(1)
