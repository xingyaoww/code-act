import os
import json
import argparse
import pathlib
from copy import deepcopy
import glob
import shutil

# load variables from config_variables.py
from mint.configs.config_variables import (
    EVALUATED_MODEL_LIST,
    FEEDBACK_PROVIDER_LIST,
    FEEDBACK_TYPES,
    ENV_CONFIGS,
    TASK_INFO_MAP,
    TASK_TYPE_TO_TOOL_IMPORT,
    FEEDBACK_CONFIG,
    DATA_OUTPUTS_DIR,
    DATASET_DIR,
)


def build_path(
    task_name: str,
    agent_model_name: str,
    feedback_type: dict,
    feedback_model_name: str,
    env_config: str,
    prefix: str,
    split: str,
):
    dir_path = prefix
    # 1. first-level dir = agent model name
    dir_path = os.path.join(dir_path, agent_model_name)

    # 2. second-level dir = feedback model name
    dir_path = os.path.join(dir_path, "F=" + feedback_model_name)
    if feedback_model_name != "None":
        # 3. (optional) third-level dir = feedback type
        pseudo_human_feedback = feedback_type["pseudo_human_feedback"]
        feedback_form = feedback_type["feedback_form"]
        dir_path = os.path.join(
            dir_path, f"PHF={pseudo_human_feedback}-{feedback_form}"
        )

    # 4. fourth-level dir = env config
    max_propose_solution = env_config["max_propose_solution"]
    max_steps = env_config["max_steps"]
    use_tools = env_config["use_tools"]
    count_down = env_config["count_down"]
    cur_dir_name = f"max{max_steps}_p{max_propose_solution}"
    if use_tools:
        cur_dir_name += "+tool"
    if count_down:
        cur_dir_name += "+cd"
    dir_path = os.path.join(dir_path, cur_dir_name)

    # 5. fifth-level dir = task type
    dir_path = os.path.join(dir_path, TASK_INFO_MAP[task_name]["type"])

    # 6. sixth-level dir = task name
    dir_path = os.path.join(dir_path, task_name)

    # 7. seventh-level dir = split
    dir_path = os.path.join(dir_path, split)
    return dir_path


def generate_config_json(
    task_name: str,
    agent_model_info: dict,
    feedback_type: dict,
    feedback_model_info: dict,
    env_config: str,
):
    model_name = agent_model_info["config"]["model_name"]

    output_filepath = build_path(
        task_name=task_name,
        agent_model_name=model_name,
        feedback_type=feedback_type,
        feedback_model_name=feedback_model_info["model_name"],
        env_config=env_config,
        prefix=DATA_OUTPUTS_DIR,
        split=TASK_INFO_MAP[task_name].get("split", "test"),
    )

    json_dict = {
        "agent": agent_model_info,
        "task": {
            "task_class": TASK_INFO_MAP[task_name]["class"],
            "task_type": TASK_INFO_MAP[task_name]["type"],
            "tool_imports": TASK_TYPE_TO_TOOL_IMPORT[TASK_INFO_MAP[task_name]["type"]],
        },
        "output_dir": output_filepath,
        "env_config": env_config,
    }
    pathlib.Path(output_filepath).mkdir(parents=True, exist_ok=True)
    with open(output_filepath + "/output.txt", "a") as f:
        pass

    if "extra_load_task_kwargs" in TASK_INFO_MAP[task_name]:
        json_dict["task"]["extra_load_task_kwargs"] = TASK_INFO_MAP[task_name][
            "extra_load_task_kwargs"
        ]

    input_filename = TASK_INFO_MAP[task_name].get("input_filename", "test_prompts.json")
    dataset_dir = TASK_INFO_MAP[task_name].get("dataset_dir", f"{DATASET_DIR}/{TASK_INFO_MAP[task_name]['type']}/")
    json_dict["task"]["filepath"] = os.path.join(dataset_dir, input_filename)
    json_dict["feedback_config"] = FEEDBACK_CONFIG
    json_dict["feedback_config"].update(feedback_type)
    json_dict["feedback_config"]["feedback_agent_config"].update(feedback_model_info)

    return json_dict


def build_json_for_all_tasks(
    agent_model_info: dict,
    feedback_type: dict,
    feedback_model_info: dict,
    env_config: dict,
):
    for task_name in TASK_INFO_MAP.keys():
        model_name = agent_model_info["config"]["model_name"]

        output_filepath = (
            build_path(
                task_name=task_name,
                agent_model_name=model_name,
                feedback_type=feedback_type,
                feedback_model_name=feedback_model_info["model_name"],
                env_config=env_config,
                prefix="configs/",
                split=TASK_INFO_MAP[task_name].get("split", "test"),
            )
            + ".json"
        )
        json_dict = generate_config_json(
            task_name=task_name,
            agent_model_info=agent_model_info,
            feedback_type=feedback_type,
            feedback_model_info=feedback_model_info,
            env_config=env_config,
        )

        pathlib.Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w") as f:
            f.write(json.dumps(json_dict, indent=4) + "\n")


def run_everything(evaluated_model_list, feedback_provider_list, env_configs):
    # With feedback
    for agent_model_info in evaluated_model_list:
        for feedback_model_info in feedback_provider_list:
            for env_config in env_configs:

                # If no feedback is provided
                if feedback_model_info["agent_class"] == "None":
                    build_json_for_all_tasks(
                        agent_model_info=agent_model_info,
                        feedback_type={
                            "pseudo_human_feedback": "None",
                            "feedback_form": "None",
                        },
                        feedback_model_info=feedback_model_info,
                        env_config=env_config,
                    )

                # If feedback is provided - generate config for all feedback types
                else:
                    # skip if not k=5
                    if env_config["max_steps"] != 5:
                        continue
                    for feedback_type in FEEDBACK_TYPES:
                        build_json_for_all_tasks(
                            agent_model_info=agent_model_info,
                            feedback_type=feedback_type,
                            feedback_model_info=feedback_model_info,
                            env_config=env_config,
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add-hard",
        action="store_true",
        help="Only generate configs for hard tasks",
    )
    args = parser.parse_args()

    # delete all existing configs
    for file in glob.glob("configs/*"):
        shutil.rmtree(file)
    
    run_everything(
        evaluated_model_list=EVALUATED_MODEL_LIST,
        feedback_provider_list=FEEDBACK_PROVIDER_LIST,
        env_configs=ENV_CONFIGS,
    )

    if args.add_hard:
        print("Adding hard tasks for GPT-4 to run")

        # Overriding the default configs
        EVALUATED_MODEL_LIST = [
            {
                "agent_class": "OpenAILMAgent",
                "config": {
                    "model_name": "gpt-4-0613",
                    "chat_mode": True,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
            },
        ]

        FEEDBACK_PROVIDER_LIST = [
            {
                "agent_class": "None",
                "model_name": "None",
            },
        ]

        ENV_CONFIGS = [
            {
                "max_steps": 5,
                "use_tools": True,
                "max_propose_solution": 2,
                "count_down": True,
            }
        ]
        # Override the dataset dir
        DATASET_DIR = "data/trajectories/hard_instances/raw"

        # modify the global config file
        for task_name, task_dict in TASK_INFO_MAP.items():
            if task_name == "alfworld":
                # only need to add extra_load_task_kwargs
                TASK_INFO_MAP[task_name]["extra_load_task_kwargs"][
                    "ids_to_run_file"
                ] = "data/trajectories/hard_instances/raw/decision_making/train/alfworld.txt"
            else:
                # override the input_filename to 
                # task_type/train/task_name.jsonl would suffice
                TASK_INFO_MAP[task_name]["input_filename"] = os.path.join(
                    "train", f"{task_name.lower()}.jsonl"
                )

        run_everything(
            evaluated_model_list=EVALUATED_MODEL_LIST,
            feedback_provider_list=FEEDBACK_PROVIDER_LIST,
            env_configs=ENV_CONFIGS,
        )
