import re
import json

def fn(**kwargs):
    return kwargs

def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

class APIParseError(Exception):
    pass

def parse_code_as_action(text):
    # model_output: code_as_action [ApiName(param1=value1, param2=value2), ...)]
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError("No valid action found in text")

    api_name = match.group(1)
    params = match.group(2)

    param_pattern = r"(\w+)\s*=\s*['\"](.+?)['\"]|(\w+)\s*=\s*(\[.*\])|(\w+)\s*=\s*(\w+)"
    param_dict = {}
    for m in re.finditer(param_pattern, params):
        if m.group(1):
            param_dict[m.group(1)] = m.group(2)
        elif m.group(3):
            param_dict[m.group(3)] = m.group(4)
        elif m.group(5):
            param_dict[m.group(5)] = m.group(6)
    return api_name, param_dict

def parse_json_as_action(text):
    # model_output: json_as_action [{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]
    pattern = r"\[(\{.*\})\]"
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError("No valid action found in text")

    json_str = match.group(1)
    try:
        json_dict = json.loads(json_str)
    except:
        raise ValueError("No valid action found in text")

    try:
        api_name = json_dict["action"]
    except KeyError:
        raise APIParseError("No valid action found. You should provide the action name with key \"action\" in json.")
    del json_dict["action"]
    param_dict = json_dict

    return api_name, param_dict

def parse_text_as_action(text):
    # model_output: text_as_action [Action: ApiName, param1: value1, param2: value2, ...]
    pattern = r"\[Action:\s*(\w+),\s*(.*)\]"
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError("No valid action found in text")

    api_name = match.group(1)
    params = match.group(2)

    param_pattern = r"(\w+):\s*([^,]+)"
    param_dict = {}
    for m in re.finditer(param_pattern, params):
        param_dict[m.group(1)] = m.group(2).strip("\"").strip("'").strip()

    return api_name, param_dict

def parse_api_call(text, action_mode):
    if action_mode == "text_as_action":
        # model_output: text_as_action [Action: ApiName, param1: value1, param2: value2, ...]
        return parse_text_as_action(text)
    elif action_mode == "json_as_action":
        # model_output: json_as_action [{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]
        return parse_json_as_action(text)
    elif action_mode == "code_as_action":
        # model_output: code_as_action [ApiName(param1=value1, param2=value2), ...)]
        return parse_code_as_action(text)
    else:
        raise NotImplementedError("action_mode {} not implemented".format(action_mode))

def convert_api_call_to_json_as_action(api_name, param_dict):
    json_dict = {"action": api_name}
    json_dict.update(param_dict)
    return "[" + json.dumps(json_dict) + "]"

def convert_api_call_to_text_as_action(api_name, param_dict):
    text = "Action: {}, ".format(api_name)
    for k, v in param_dict.items():
        text += "{}: {}, ".format(k, v)
    return "[" + text[:-2] + "]" # remove last ", "

def convert_api_call_to_code_as_action(api_name, param_dict):
    text = "{}(".format(api_name)
    for k, v in param_dict.items():
        text += "{}=\'{}\', ".format(k, v)
    text = text[:-2] + ")"  # remove last ", "
    return "[" + text + "]"
