import os
import yaml
import logging
from typing import Iterable
import alfworld
import alfworld.agents.environment as envs
import json
import os

LOGGER = logging.getLogger("MINT")

from mint.tasks.base import Task

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        id: str,
        prompt: str,
        reference: str,
        env: envs.AlfredTWEnv,
        task_type: str,
        **kwargs,
    ):
        self.task_name = f"alfworld/{task_type}"  # used to load the correct ICL example
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt.strip()
        self._reference = reference
        self.metadata["task_type"] = task_type

        # NOTE: AlfWorld is different -> no reference solution
        self._env = env

    @property
    def env(self) -> envs.AlfredTWEnv:
        """Stateful environment of the task.

        Specific for AlfWorld.
        """
        return self._env

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task."""
        # Task success is checked at the environment level, not at the solution level.
        raise NotImplementedError

    @classmethod
    def load_tasks(cls, path: str = "./data/raw/alfworld", **kwargs) -> Iterable["Task"]:
        """Load alfworld data and prompts from a directory."""

        with open(os.path.join(path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)

        # Split following ReAct
        # https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
        SPLIT_TO_N_TASKS = {
            "eval_out_of_distribution": 134,
            "train": 3553
        }

        split = kwargs.get("split", "eval_out_of_distribution")
        LOGGER.info(f"Loading {split} split of AlfWorld data.")

        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=1)

        N_TASKS = SPLIT_TO_N_TASKS[split]

        ids_to_run_file = kwargs.get("ids_to_run_file", None)
        if ids_to_run_file is not None:
            with open(ids_to_run_file) as f:
                ids_to_run = set(f.read().splitlines())
            LOGGER.info(f"Running only {len(ids_to_run)} tasks.")
        else:
            ids_to_run = None


        def generator():

            for _ in range(N_TASKS):
                ob, info = env.reset()
                ob = "\n".join(ob[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                if split != "train":
                    # Load ground truth reference for evaluation
                    # This is only used for +GT ablation
                    gt_reference_file = os.path.join(
                        os.path.dirname(game_file), "gt_traj.txt"
                    )

                    with open(gt_reference_file) as f:
                        gt_reference = f.read()
                else:
                    gt_reference = None

                name = "/".join(game_file.split("/")[-3:-1])
                if ids_to_run is not None and name not in ids_to_run:
                    LOGGER.info(f"Skipping task {name} from {split} split since not in ids_to_run.")
                    continue
                else:
                    LOGGER.info(f"Loaded task {name} from {split} split.")

                task_type = None
                for _, (k, v) in enumerate(PREFIXES.items()):
                    if name.startswith(k):
                        task_type = v
                        break
                assert task_type is not None, f"Task type not found for {name}"

                prompt = "Interact with a household to solve a task. \n" + ob
                yield cls(
                    id=name,
                    prompt=prompt,
                    reference=gt_reference,
                    env=env,
                    task_type=task_type,
                    loaded_history=None,
                )

        return generator(), N_TASKS
