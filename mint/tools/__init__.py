from .base import Tool
from typing import List
import warnings

warnings.filterwarnings("ignore")


def get_toolset_description(tools: List[Tool]) -> str:
    if len(tools) == 0:
        return ""

    output = "Tool function available (already imported in <execute> environment):\n"
    for i, tool in enumerate(tools):
        output += f"[{i + 1}] {tool.signature}\n"
        output += f"{tool.description}\n"

    return output
