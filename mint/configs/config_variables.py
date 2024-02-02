# all configs are flexible to adapt with this config_variables.py
# if you want to change data output dir or dataset dir, please change DATA_OUTPUTS_DIR and DATASET_DIR
# if you want to add more configs, please:
# (1) if you want to add tools, please add to TASK_TYPE_TO_TOOL_IMPORT
# (2) if you want to add tasks, please add to TASK_INFO_MAP
# (3) if you want to add agent models, please add to agent_model_infos
# (4) if you want to add feedback models and configs, please add to feedback_model_infos and feedback_types
# (5) if you want to add env configs, please add to env_configs

DATA_OUTPUTS_DIR = "data/outputs"
DATASET_DIR = "data/processed"

TASK_TYPE_TO_TOOL_IMPORT = {
    "reasoning": [("mint.tools.wikipedia_search", "WikipediaQueryRun")],
    "decision_making": [],
    "code_generation": [],
    "tabular": [],
}

TASK_INFO_MAP = {
    # === Reasoning ===
    "algebra": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/algebra.jsonl",
        "split": "train"
    },
    "counting_and_probability": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/counting_and_probability.jsonl",
        "split": "train"
    },
    "geometry": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/geometry.jsonl",
        "split": "train"
    },
    "intermediate_algebra": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/intermediate_algebra.jsonl",
        "split": "train"
    },
    "number_theory": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/number_theory.jsonl",
        "split": "train"
    },
    "prealgebra": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/prealgebra.jsonl",
        "split": "train"
    },
    "precalculus": {
        "class": "MATHTask", "type": "reasoning",
        "input_filename": "train/precalculus.jsonl",
        "split": "train"
    },

    # "theoremqa": {"class": "TheoremqaTask", "type": "reasoning"},
    # "gsm8k": {"class": "ReasoningTask", "type": "reasoning"},
    "hotpotqa": {
        "class": "ReasoningTask",
        "type": "reasoning",
        "input_filename": "train/hotpotqa.jsonl",
        "split": "train"
    },

    "strategyqa": {
        "class": "ReasoningTask",
        "type": "reasoning",
        "input_filename": "train/strategyqa.jsonl",
        "split": "train"
    },
    # "mmlu": {"class": "MultipleChoiceTask", "type": "reasoning"},
    # "math": {"class": "ReasoningTask", "type": "reasoning"},
    
    # # === Code Generation ===
    # "mbpp": {
    #     "class": "MBPPTask",
    #     "type": "code_generation",
    # },
    # "humaneval": {
    #     "class": "HumanEvalTask",
    #     "type": "code_generation",
    # },
    "APPS": {
        "class": "APPSTask",
        "type": "code_generation",
        "input_filename": "train/apps.jsonl",
        "split": "train"
    },
    # === Decision Making ===
    "alfworld": {
        "class": "AlfWorldTask",
        "type": "decision_making",
        "split": "train",
        "input_filename": "",  # Empth string since it needs a directory
        "extra_load_task_kwargs" : {
            "split": "train"
        },
        "dataset_dir": "data/raw/alfworld",
    },

    # === Tabular ===
    "wiki_table_questions": {
        "class": "WikiTableQuestionsTask",
        "type": "tabular",
        "input_filename": "train/wiki_table_questions.jsonl",
        "split": "train"
    },
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
        "agent_class": "OpenAILMAgent",
        "config": {
            "model_name": "gpt-3.5-turbo-0613",
            "chat_mode": True,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    },
    {
        "agent_class": "OpenAILMAgent",
        "config": {
            "model_name": "gpt-3.5-turbo-16k-0613",
            "chat_mode": True,
            "max_tokens": 1024,
            "temperature": 0.0,
        },
    },
    {
        "agent_class": "ClaudeLMAgent",
        "config": {
            "model_name": "claude-2",
            "chat_mode": True,
            "max_tokens": 1024,
            "temperature": 0.0,
        },
    },
    {
        "agent_class": "ClaudeLMAgent",
        "config": {
            "model_name": "claude-instant-1",
            "chat_mode": True,
            "max_tokens": 1024,
            "temperature": 0.0,
        },
    },
    # {
    #     "agent_class": "VLLMAgent",
    #     "config": {
    #         "model_name": "Mistral-7B-v0.1",
    #         "chat_mode": False,
    #         "max_tokens": 1024,
    #         "temperature": 0.0,
    #         "openai.api_base": "http://localhost:8010/v1",
    #         "add_system_message": False,
    #     },
    # },
]

FEEDBACK_PROVIDER_LIST = [
    {
        "agent_class": "None",
        "model_name": "None",
    },
    # {
    #     "agent_class": "OpenAIFeedbackAgent",
    #     "model_name": "gpt-4-0613",
    # },
]

FEEDBACK_TYPES = [
    # {"pseudo_human_feedback": "no_GT", "feedback_form": "textual"},
    # {"pseudo_human_feedback": "no_GT", "feedback_form": "binary"},
    # {"pseudo_human_feedback": "GT", "feedback_form": "binary"},
    # {"pseudo_human_feedback": "GT", "feedback_form": "textual"},
]

ENV_CONFIGS = [
    # {
    #     "max_steps": 10,
    #     "use_tools": True,
    #     "max_propose_solution": 2,
    #     "count_down": True,
    # },
    {
        "max_steps": 5,
        "use_tools": True,
        "max_propose_solution": 2,
        "count_down": True,
    }
]
