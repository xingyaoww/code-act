from typing import Optional

# Based on Table 3 of the original paper
# https://arxiv.org/pdf/2203.07540.pdf

def open_container(container: str) -> str:
    """Open a container."""
    return f"open {container.strip()}"

def close_container(container: str) -> str:
    """Close a container."""
    return f"close {container.strip()}"

def activate(device: str) -> str:
    """Activate a device."""
    return f"activate {device.strip()}"

def deactivate(device: str) -> str:
    """Deactivate a device."""
    return f"deactivate {device.strip()}"

def connect_obj1_to_obj2(obj1: str, obj2: str) -> str:
    """Connect electrical components."""
    return f"connect {obj1.strip()} to {obj2.strip()}"

def disconnect_obj(obj: str) -> str:
    """Disconnect electrical components."""
    return f"disconnect {obj.strip()}"

def use(item: str, target: Optional[str] = None) -> str:
    """Use a device or item (optionally on a target)."""
    if target:
        return f"use {item.strip()} on {target.strip()}"
    else:
        return f"use {item.strip()}"

def look_around() -> str:
    """Look around the room and receive a description of the current room."""
    return "look around"

def look_at(obj: str) -> str:
    """Describe an object in detail."""
    return f"look at {obj.strip()}"

def look_in(container: str) -> str:
    """Describe a container’s contents."""
    return f"look in {container.strip()}"

def read(obj: str) -> str:
    """Read a note or book."""
    return f"read {obj.strip()}"

def move_to(obj: str, container: str) -> str:
    """Move an object to a container."""
    return f"move {obj.strip()} to {container.strip()}"

def pick_up(obj: str) -> str:
    """Move an object to the inventory."""
    return f"pick up {obj.strip()}"

def put_down(obj: str) -> str:
    """Drop an inventory item."""
    return f"put down {obj.strip()}"

def pour_into(liquid: str, container: str) -> str:
    """Pour a liquid into a container."""
    return f"pour {liquid.strip()} into {container.strip()}"

def dunk_into(container: str, liquid: str) -> str:
    """Dunk a container into a liquid."""
    return f"dunk {container.strip()} into {liquid.strip()}"

def mix(container: str) -> str:
    """Chemically mix a container."""
    return f"mix {container.strip()}"

def go_to(location: str) -> str:
    """Move to a new location."""
    return f"go to {location.strip()}"

def teleport_to(location: str) -> str:
    """Teleport to a specific room."""
    return f"teleport to {location.strip()}"

def eat(item: str) -> str:
    """Eat a food."""
    return f"eat {item.strip()}"

def drop(obj_a: str, obj_b: Optional[str] = None) -> str:
    if obj_b is None:
        return f"drop {obj_a.strip()}"
    return f"drop {obj_a.strip()} to {obj_b.strip()}"


def flush(item: str) -> str:
    """Flush a toilet."""
    return f"flush {item.strip()}"

def focus_on(item: str) -> str:
    """Signal intent on a task object."""
    return f"focus on {item.strip()}"

def wait(duration: Optional[int] = None) -> str:
    """Take no action for some duration."""
    if duration is not None:
        return f"wait{duration}"
    else:
        return "wait"

def task() -> str:
    """Describe the current task."""
    return "task"

def inventory() -> str:
    """List agent's inventory."""
    return "inventory"

def examine(obj: str) -> str:
    """Describe an object in detail. Look at an object carefully. For example, examine(\"apple\"). Note that you cannot examine a location."""
    return f"examine {obj.strip()}"


DEFAULT_USER_NS = {
    "open_container": open_container,
    "close_container": close_container,
    "activate": activate,
    "deactivate": deactivate,
    "connect_obj1_to_obj2": connect_obj1_to_obj2,
    "disconnect_obj": disconnect_obj,
    "use": use,
    "look_around": look_around,
    "look_at": look_at,
    "look_in": look_in,
    "read": read,
    "move_to": move_to,
    "pick_up": pick_up,
    "put_down": put_down,
    "pour_into": pour_into,
    "dunk_into": dunk_into,
    "mix": mix,
    "go_to": go_to,
    "teleport_to": teleport_to,
    "eat": eat,
    "drop": drop,
    "flush": flush,
    "focus_on": focus_on,
    "wait": wait,
    "task": task,
    "inventory": inventory,
    "examine": examine,
}

from typing import Mapping
import re
import signal
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any

class PythonREPL:
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
        self,
        user_ns: Mapping[str, Any] = DEFAULT_USER_NS,
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
                r"File (.*)mint/tools/python_tool.py:(\d+)",
                r"File <hidden_filepath>:\1",
                output,
            )
            if len(output) > 2000:
                output = output[:2000] + "...\n[Output Truncated]"

        return output
