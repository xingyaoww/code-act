from mint.tools.base import Tool
from mint.datatypes import StepOutput


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


class AlfWorldTool(Tool):
    """Base class for a tool in AlfWorld."""

    name = ""
    signature = ""
    description = ""

    def __init__(self, env, callback_fn: callable) -> None:
        self.env = env
        self.callback_fn = callback_fn

    def __call__(self, **kwargs) -> StepOutput:
        pass

    def step(self, action: str) -> str:
        # Follow ReAct Implementation
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
        self.callback_fn(
            StepOutput(
                observation=observation,
                success=done,
                extra={"reward": reward},
            )
        )
        # print(f"After '{action}': {observation}")
        return observation


SIGNATURE_TEMPLATE = "{name}({args}) -> str"

DESCRIPTION_TEMPLATE = """{desc}
For example: {example}
""".lstrip()


class Put(AlfWorldTool):
    name = "put"
    signature = SIGNATURE_TEMPLATE.format(
        name=name, args="object: str, receptacle: str"
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Put an object in/on a receptacle.",
        example='put("mug 1", "desk 2")',
    )

    def __call__(self, object: str, receptacle: str):
        return self.step(f"put {object} in/on {receptacle}")


class GoTo(AlfWorldTool):
    name = "goto"
    signature = SIGNATURE_TEMPLATE.format(name=name, args="receptacle: str")
    description = DESCRIPTION_TEMPLATE.format(
        desc="Go to a location of the receptacle.",
        example='goto("drawer 1")',
    )

    def __call__(self, receptacle: str):
        return self.step(f"go to {receptacle}")


class Take(AlfWorldTool):
    name = "take_from"
    signature = SIGNATURE_TEMPLATE.format(
        name=name, args="object: str, receptacle: str"
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Take an object from a receptacle.",
        example='take_from("mug 1", "shelf 2")',
    )

    def __call__(self, object: str, receptacle: str):
        return self.step(f"take {object} from {receptacle}")


class Open(AlfWorldTool):
    name = "open_receptacle"
    signature = SIGNATURE_TEMPLATE.format(name=name, args="receptacle: str")
    description = DESCRIPTION_TEMPLATE.format(
        desc="Open a receptacle.",
        example='open_receptacle("fridge 1")',
    )

    def __call__(self, receptacle: str):
        return self.step(f"open {receptacle}")


class Toggle(AlfWorldTool):
    name = "toggle"
    signature = SIGNATURE_TEMPLATE.format(name=name, args="object_or_receptacle: str")
    description = DESCRIPTION_TEMPLATE.format(
        desc="Toggle an object or receptacle.",
        example='toggle("light 2")',
    )

    def __call__(self, object_or_receptacle: str):
        return self.step(f"toggle {object_or_receptacle}")


class Close(AlfWorldTool):
    name = "close_receptacle"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="receptacle: str",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Close a receptacle.",
        example='close_receptacle("microwave 1")',
    )

    def __call__(self, receptacle: str):
        return self.step(f"close {receptacle}")


class Clean(AlfWorldTool):
    name = "clean"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="object: str, receptacle: str",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Clean an object with a receptacle.",
        example='clean("cloth 1", "sinkbasin 1")',
    )

    def __call__(self, object: str, receptacle: str):
        return self.step(f"clean {object} with {receptacle}")


class Heat(AlfWorldTool):
    name = "heat"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="object: str, receptacle: str",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Heat an object with a receptacle.",
        example='heat("egg 1", "microwave 1")',
    )

    def __call__(self, object: str, receptacle: str):
        return self.step(f"heat {object} with {receptacle}")


class Cool(AlfWorldTool):
    name = "cool"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="object: str, receptacle: str",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Cool an object with a receptacle.",
        example='cool("bottle 1", "fridge 1")',
    )

    def __call__(self, object: str, receptacle: str):
        return self.step(f"cool {object} with {receptacle}")


# add use tool
class Use(AlfWorldTool):
    name = "use"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="receptacle: str",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Use a receptacle.",
        example='use("lamp 1")',
    )

    def __call__(self, receptacle: str):
        return self.step(f"use {receptacle}")


# add look tool
class Look(AlfWorldTool):
    name = "look"
    signature = SIGNATURE_TEMPLATE.format(
        name=name,
        args="",
    )
    description = DESCRIPTION_TEMPLATE.format(
        desc="Look around. It will return what you see in the room.",
        example="look()",
    )

    def __call__(self):
        return self.step(f"look")


ALFWORLD_TOOL_CLS = [
    Put,
    GoTo,
    Take,
    Open,
    Toggle,
    Close,
    Clean,
    Heat,
    Cool,
    Use,
    Look,
]
