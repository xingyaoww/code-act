from typing import Mapping
from mint.tools.base import Tool
import re
import signal
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any

class PythonREPL(Tool):
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
        self,
        user_ns: Mapping[str, Any],
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.user_ns = user_ns
        self.timeout = timeout
        self.reset()

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    def reset(self) -> None:
        InteractiveShell.clear_instance()
        self.shell = InteractiveShell.instance(
            # NOTE: shallow copy is needed to avoid
            # shell modifying the original user_ns dict
            user_ns=dict(self.user_ns),
            colors="NoColor",
        )

    def __call__(self, query: str) -> str:
        """Use the tool and return observation"""
        with self.time_limit(self.timeout):
            # NOTE: The timeout error will be caught by the InteractiveShell

            # Capture all output
            with io.capture_output() as captured:
                _ = self.shell.run_cell(query, store_history=True)
            output = captured.stdout

            if output == "":
                output = "[Executed Successfully with No Output]"

            # replace potentially sensitive filepath
            # e.g., File /mint/mint/tools/python_tool.py:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
            # with File <filepath>:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
            # use re
            output = re.sub(
                # r"File (/mint/)mint/tools/python_tool.py:(\d+)",
                r"File (.*)mint/tools/python_tool.py:(\d+)",
                r"File <hidden_filepath>:\1",
                output,
            )
            if len(output) > 2000:
                output = output[:2000] + "...\n[Output Truncated]"

        return output


if __name__ == "__main__":
    shell = PythonREPL(
        user_ns={"hello": lambda: print("hello world")},
        timeout=5,
    )

    def exec_tool(input_code: str, cur_shell=shell) -> str:
        print("=== INPUT ===")
        print(input_code)
        print("=== OUTPUT ===")
        output = cur_shell(input_code)
        print(output)

    INPUT = """
import time
i = 0
while True:
    time.sleep(1)
    i += 1
    print(i)
"""
    exec_tool(INPUT)

    INPUT = """
a = 1
b = 2
a + b
"""
    exec_tool(INPUT)

    # test whether variables `a` persists
    INPUT = """a + 7"""
    exec_tool(INPUT)

    # test provided import
    INPUT = """hello()"""
    exec_tool(INPUT)

    print("\n=== SECOND SHELL ===\n")
    second_shell = PythonREPL(
        user_ns={"hello": lambda: print("not hello world!")},
        timeout=5,
    )

    INPUT = """a + 7"""
    exec_tool(INPUT, cur_shell=second_shell)

    INPUT = """hello()"""
    exec_tool(INPUT, cur_shell=second_shell)

    # reset tool and test whether variables `a` persists
    shell.reset()

    print("\n=== 1st Shell RESET ===\n")

    INPUT = """a + 7"""
    exec_tool(INPUT)

    # test package import
    INPUT = """
import math
math.sqrt(4)
"""
    exec_tool(INPUT)

    # test provided import after reset
    INPUT = """hello()"""
    exec_tool(INPUT)

    # code that leads to traceback
    INPUT = """
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) | set(test_tup2))
    return res
res = similar_elements((3, 4, 5, 6), (5, 7, 4, 10))
assert res == (4, 5), "Expected (4, 5) but got {}".format(res)
"""
    exec_tool(INPUT)

    # code with no error
    INPUT = """
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
res = similar_elements((3, 4, 5, 6), (5, 7, 4, 10))
assert res == (4, 5), "Expected (4, 5) but got {}".format(res)
"""
    exec_tool(INPUT)
