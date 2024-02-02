# all configs are flexible to adapt with this config_variables.py
# if you want to change data output dir or dataset dir, please change DATA_OUTPUTS_DIR and DATASET_DIR
# if you want to add more configs, please:
# (1) if you want to add tools, please add to TASK_TYPE_TO_TOOL_IMPORT
# (2) if you want to add tasks, please add to TASK_INFO_MAP
# (3) if you want to add agent models, please add to agent_model_infos
# (4) if you want to add feedback models and configs, please add to feedback_model_infos and feedback_types
# (5) if you want to add env configs, please add to env_configs
import os
DATA_OUTPUTS_DIR = "data/outputs"
DATASET_DIR = "data/processed"

TASK_TYPE_TO_TOOL_IMPORT = {
    "reasoning": [("mint.tools.wikipedia_search", "WikipediaQueryRun")],
    "decision_making": [],
    "code_generation": [],
}

TASK_INFO_MAP = {
    # === Reasoning ===
    "theoremqa": {"class": "TheoremqaTask", "type": "reasoning"},
    "gsm8k": {"class": "ReasoningTask", "type": "reasoning"},
    "hotpotqa": {"class": "ReasoningTask", "type": "reasoning"},
    "mmlu": {"class": "MultipleChoiceTask", "type": "reasoning"},
    "math": {"class": "ReasoningTask", "type": "reasoning"},
    # === Code Generation ===
    "mbpp": {
        "class": "MBPPTask",
        "type": "code_generation",
    },
    "humaneval": {
        "class": "HumanEvalTask",
        "type": "code_generation",
    },
    # === Decision Making ===
    "alfworld": {"class": "AlfWorldTask", "type": "decision_making"},
}

FEEDBACK_CONFIG = {
    "feedback_agent_config": {
        "chat_mode": True,
        "max_tokens": 1024,
        "temperature": 0.0,
        "stop": ["\nQ:"],
    }
}

EVALUATED_MODEL_LIST = [
    {
        "agent_class": "VLLMAgent",
        "config": {
            "model_name": "code-act-agent",
            "chat_mode": True,
            "max_tokens": 512,
            "temperature": 0.0,
            "openai.api_base": os.getenv("OPENAI_API_BASE"),
            "add_system_message": False,
        },
    }
]

FEEDBACK_PROVIDER_LIST = [
    {
        "agent_class": "None",
        "model_name": "None",
    },
    {
        "agent_class": "OpenAIFeedbackAgent",
        "model_name": "gpt-4-0613",
    },
]

FEEDBACK_TYPES = [
    {"pseudo_human_feedback": "no_GT", "feedback_form": "textual"},
]

ENV_CONFIGS = [
    {
        "max_steps": 5,
        "use_tools": True,
        "max_propose_solution": 2,
        "count_down": True,
    },
    {
        "max_steps": 4,
        "use_tools": True,
        "max_propose_solution": 2,
        "count_down": True,
    },
    {
        "max_steps": 3,
        "use_tools": True,
        "max_propose_solution": 2,
        "count_down": True,
    },
    {
        "max_steps": 2,
        "use_tools": True,
        "max_propose_solution": 2,
        "count_down": True,
    },
    {
        "max_steps": 1,
        "use_tools": True,
        "max_propose_solution": 1,
        "count_down": True,
    }
]
