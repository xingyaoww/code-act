import json
from tool_manager import ToolManager
import re
from rouge import Rouge
import os
from utils import OpenAIChatWrapper, ClaudeWrapper, DavinciWrapper, GeminiWrapper
import logging
from tqdm import tqdm
from api_call_extraction import (
    APIParseError,
    parse_api_call,
    convert_api_call_to_json_as_action,
    convert_api_call_to_text_as_action,
    convert_api_call_to_code_as_action,
)
from datetime import datetime
import numpy as np

from termcolor import colored

def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

class Sample:
    def __init__(self, chat_history, apis, ground_truth):
        self.chat_history = chat_history
        self.apis = apis
        self.ground_truth = ground_truth

    def __repr__(self):
        return 'Sample(chat_history={}, apis={}, ground_truth={})'.format(self.chat_history, self.apis, self.ground_truth)
        # return 'Chat history: {}, apis: {}, ground truth: {}'.format(self.chat_history, self.apis, self.ground_truth)

    @classmethod
    def from_chat_history(cls, chat_history):
        apis = set()
        api_positions = []
        for i, item in enumerate(chat_history):
            if item['role'] == 'API':
                apis.add(item['api_name']) 
                api_positions.append(i)

        samples = []
        for i in api_positions:
            sample = cls(chat_history[:i], apis, chat_history[i])
            samples.append(sample)
            sample = cls(chat_history[:i + 1], apis, chat_history[i + 1])
            samples.append(sample)

        return samples


class Evaluator:

    def __init__(self, samples):
        self.dataset = samples
        self.sample_ids = list(range(len(self.dataset)))

    def get_all_sample_ids(self):
        return self.sample_ids
    
    def get_api_description(self, api_name):
        tool_manager = ToolManager()
        return tool_manager.get_api_description(api_name)

    
    def get_model_input(self, sample_id):
        sample = self.dataset[sample_id]
        apis = sample.apis
        chat_history = sample.chat_history
        tool_manager = ToolManager()
        api_descriptions = []
        for api_name in apis:
            api_descriptions.append(tool_manager.get_api_description(api_name))
        api_descriptions = '\n'.join(api_descriptions)
        return api_descriptions, chat_history

    
    def evaluate(self, sample_id, model_output, action_mode):
        assert action_mode in ['text_as_action', 'json_as_action', 'code_as_action']
        # model_output: text_as_action [Action: ApiName, param1: value1, param2: value2, ...]
        # model_output: json_as_action [{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]
        # model_output: code_as_action [ApiName(param1=value1, param2=value2), ...)]
        tool_manager = ToolManager()

        sample = self.dataset[sample_id]
        ground_truth = sample.ground_truth
        if ground_truth['role'] == 'API':
            logging.info(f"** Model output: {colored(model_output, 'yellow')}")
            try:
                api_name, param_dict = parse_api_call(model_output, action_mode)
            except APIParseError as e:
                return False, str(e)
            logging.info(f"** Parsed API call: {colored(api_name, 'green')}, {colored(param_dict, 'green')}")
            if api_name != ground_truth['api_name']:
                return False, 'API Name Mismatch: {} vs {}'.format(api_name, ground_truth['api_name'])
            try:
                result = tool_manager.api_call(api_name, **param_dict)
            except Exception as e:
                return False, str(e)
            api = tool_manager.init_tool(api_name)
            try:
                correct = api.check_api_call_correctness(result, ground_truth['result'])
            except KeyError:
                correct = False
                result = 'KeyError' + str(result)
            return correct, result
        elif ground_truth['role'] == 'AI':
            score = calculate_rouge_l_score(ground_truth['text'], model_output)
            return round(score, 4)
        

def get_api_call(model_output):
    # api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = r"\[(.*)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

BASE_CALL_PROMPT = '''Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of {format_guide}, replace the ApiName with the actual API name, and replace the key and value with the actual parameters.
Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output. 
This year is 2023.
Input: 
User: [User's utterence]
AI: [AI's utterence]

Expected output:
{expected_output}

API descriptions:
'''

ACTION_MODE_TO_CALL_PROMPT = {
    "code_as_action": BASE_CALL_PROMPT.format(
        format_guide="[ApiName(key1='value1', key2='value2', ...)]",
        expected_output="[ApiName(key1='value1', key2='value2', ...)]"
    ),
    "text_as_action": BASE_CALL_PROMPT.format(
        format_guide='[Action: ApiName, key1: value1, key2: value2, ...]',
        expected_output='[Action: ApiName, key1: value1, key2: value2, ...]'
    ),
    "json_as_action": BASE_CALL_PROMPT.format(
        format_guide='[{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]',
        expected_output='[{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]'
    )
}

BASE_RESPONSE_PROMPT = '''Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
This year is 2023.
Input: 
User: [User's utterence]
AI: [AI's utterence]
{expected_output}

Expected output:
AI: [AI's utterence]

API descriptions:
'''

ACTION_MODE_TO_RESPONSE_PROMPT = {
    "code_as_action": BASE_RESPONSE_PROMPT.format(
        expected_output="[ApiName(key1='value1', key2='value2', ...)]"
    ),
    "text_as_action": BASE_RESPONSE_PROMPT.format(
        expected_output='[Action: ApiName, key1: value1, key2: value2, ...]'
    ),
    "json_as_action": BASE_RESPONSE_PROMPT.format(
        expected_output='[{"action": "ApiName", "param1": "value1", "param2": "value2", ...}]'
    )
}

def combine_assistant_messages(messages):
    # if there's to consecutive assistant messages, combine them
    combined_messages = []
    for message in messages:
        if message['role'] == 'assistant' \
            and combined_messages \
            and combined_messages[-1]['role'] == 'assistant':
            combined_messages[-1]['content'] += '\n' + message['content']
        else:
            combined_messages.append(message)
    return combined_messages

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='lv1-lv2-samples/level-1-given-desc')
    parser.add_argument('--action_mode', default="json_as_action")
    parser.add_argument('--api_test_enabled', default=False, action='store_true')
    parser.add_argument('--dialog_test_enabled', default=False, action='store_true')
    parser.add_argument('--model_name', default='gpt-3.5-turbo-0613')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    
    assert args.action_mode in ['text_as_action', 'json_as_action', 'code_as_action']
    api_call_prompt = ACTION_MODE_TO_CALL_PROMPT[args.action_mode]
    response_prompt = ACTION_MODE_TO_RESPONSE_PROMPT[args.action_mode]

    data_dir = args.data_dir
    # api_test_enabled = False
    api_test_enabled = args.api_test_enabled
    # dialog_test_enabled = not api_test_enabled
    dialog_test_enabled = args.dialog_test_enabled
    assert api_test_enabled or (not api_test_enabled and dialog_test_enabled), 'Either api_test_enabled or dialog_test_enabled should be True'

    api_str = 'api' if api_test_enabled else 'response'

    dataset = data_dir.split('/')[-1]
    output_dir = os.path.join(
        args.output_dir,
        dataset,
        args.model_name,
        args.action_mode
    )
    # redirect stdout and stderr to log file
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, 'eval-{}-{}.log'.format(api_str, datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    state_file = os.path.join(
        output_dir, 'eval-{}-state.json'.format(api_str)
    )
    output_results_json = os.path.join(
        output_dir, '{}-results.json'.format(api_str)
    )
    if os.path.exists(output_results_json):
        print('Results already exist: {}'.format(output_results_json))
        exit(0)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info('Data dir: {}'.format(data_dir))
    logging.info('Output dir: {}'.format(output_dir))
    logging.info('API test enabled: {}'.format(api_test_enabled))
    logging.info('Dialog test enabled: {}'.format(dialog_test_enabled))
    logging.info('Action mode: {}'.format(args.action_mode))

    if args.action_mode == 'code_as_action':
        api_convert_fn = convert_api_call_to_code_as_action
    elif args.action_mode == 'text_as_action':
        api_convert_fn = convert_api_call_to_text_as_action
    elif args.action_mode == 'json_as_action':
        api_convert_fn = convert_api_call_to_json_as_action

    if os.path.basename(data_dir).endswith('given-desc'):
        tool_search_enabled = False
    else:
        tool_search_enabled = True

    if "claude" in args.model_name:
        llm = ClaudeWrapper(model_name=args.model_name)
    elif "davinci" in args.model_name:
        llm = DavinciWrapper(model_name=args.model_name)
    elif "gemini" in args.model_name:
        llm = GeminiWrapper(model_name=args.model_name)
    else:
        llm = OpenAIChatWrapper(model_name=args.model_name)

    state = {
        'total_api_calls': 0,
        'correct_api_calls': 0,
        'rougel_scores': [],
        'processed_file_ids': set()
    }
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        state["processed_file_ids"] = set(state["processed_file_ids"])

    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

    for file in tqdm(jsonl_files, desc='Processing files', ncols=100):
        history = []
        with open(os.path.join(data_dir, file), 'r') as f:
            for line in f:
                history.append(json.loads(line))
        samples = Sample.from_chat_history(history)
        evaluator = Evaluator(samples)

        for sample_id in evaluator.get_all_sample_ids():
            file_sample_id = '{}-{}'.format(file, sample_id)
            if file_sample_id in state['processed_file_ids']:
                # skip processed samples
                continue

            sample = evaluator.dataset[sample_id]

            if sample.ground_truth['role'] == 'API' and api_test_enabled:
                if tool_search_enabled:
                    _, chat_history = evaluator.get_model_input(sample_id)
                    api_descriptions = evaluator.get_api_description('ToolSearcher')
                else:
                    api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                prompt = api_call_prompt + api_descriptions
                messages = [
                    {'role': 'system', 'content': prompt},
                ]
                for item in chat_history:
                    if item['role'] == 'User':
                        chat_role = 'user'
                        chat_content = item['text']
                        messages.append({'role': chat_role, 'content': chat_content})
                    elif item['role'] == 'AI':
                        chat_role = 'assistant'
                        chat_content = item['text']
                        messages.append({'role': chat_role, 'content': chat_content})
                    elif item['role'] == 'API':
                        api_call_str = api_convert_fn(
                            item['api_name'],
                            item['param_dict']
                        )
                        messages.append({'role': 'assistant', 'content': api_call_str})
                        messages.append({'role': 'user', 'content': f"Response: {str(item['result']['output'])}"})
                    else:
                        raise ValueError('Invalid chat role: {}'.format(item['role']))
                
                messages = combine_assistant_messages(messages)
                response = llm.call(messages)
                
                # Flatten the response
                _to_print = ""
                for message in messages:
                    _to_print += "\n" + message['role'] + ": " + message['content'] + "\n"
                logging.info(colored(_to_print, 'blue'))

                if response is None:
                    # likely an error, skip this sample
                    correct = False
                    model_output = 'Context Exceeded'
                    model_output_result = 'None'
                else:
                    if isinstance(llm, OpenAIChatWrapper):
                        model_output = response['choices'][0]['message']['content']
                    elif isinstance(llm, ClaudeWrapper):
                        model_output = response
                    elif isinstance(llm, GeminiWrapper):
                        model_output = response
                    else:
                        assert isinstance(llm, DavinciWrapper)
                        model_output = response['choices'][0]['text'].strip()

                    api_call = get_api_call(model_output)
                    if api_call:
                        try:
                            correct, model_output_result = evaluator.evaluate(
                                sample_id,
                                api_call,
                                action_mode=args.action_mode
                            )
                        except AssertionError as e:
                            if not 'The API name is not correct.' in str(e):
                                raise e
                            logging.info('AssertionError: {}'.format(e))
                            correct = False
                        except ValueError as e:
                            # unparseable api call
                            if not 'No valid action found in text' in str(e):
                                raise e
                            logging.info('ValueError: {}'.format(e))
                            correct = False
                    else:
                        model_output_result = 'No API call found'
                        correct = False
                
                if correct:
                    # correct_api_calls += 1
                    state['correct_api_calls'] += 1
                    logging.info('Correct API call: {}\nGround truth: {}'.format(api_call, sample.ground_truth))
                else:                    
                    logging.info('Incorrect model output: {}\nResult: {}\nGround truth: {}\nFile: {}\nSample ID: {}\nMessages: {}'.format(
                        colored(model_output.replace('\n', ' '), 'red'),
                        model_output_result,
                        sample.ground_truth,
                        file,
                        sample_id,
                        messages[1:]
                    ))
                # total_api_calls += 1
                state['total_api_calls'] += 1

            elif sample.ground_truth['role'] == 'AI' and dialog_test_enabled:
                api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                prompt = response_prompt + api_descriptions
                messages = [
                    {'role': 'system', 'content': prompt},
                ]
                for item in chat_history:
                    if item['role'] == 'User':
                        chat_role = 'user'
                        chat_content = item['text']
                        messages.append({'role': chat_role, 'content': chat_content})
                    elif item['role'] == 'AI':
                        chat_role = 'assistant'
                        chat_content = item['text']
                        messages.append({'role': chat_role, 'content': chat_content})
                    elif item['role'] == 'API':
                        api_call_str = api_convert_fn(
                            item['api_name'],
                            item['param_dict']
                        )
                        messages.append({'role': 'assistant', 'content': api_call_str})
                        messages.append({'role': 'user', 'content': f"Response: {str(item['result']['output'])}"})
                    else:
                        raise ValueError('Invalid chat role: {}'.format(item['role']))
                
                messages = combine_assistant_messages(messages)
                response = llm.call(messages)

                if response is None:
                    # likely an error, skip this sample
                    score = 0
                else:
                    if isinstance(llm, OpenAIChatWrapper):
                        model_output = response['choices'][0]['message']['content']
                    elif isinstance(llm, ClaudeWrapper):
                        model_output = response
                    elif isinstance(llm, GeminiWrapper):
                        model_output = response
                    else:
                        assert isinstance(llm, DavinciWrapper)
                        model_output = response['choices'][0]['text'].strip()

                    if model_output:
                        score = evaluator.evaluate(sample_id, model_output, action_mode=args.action_mode)
                    else:
                        score = 0    
                # rougel_scores.append(score)
                state['rougel_scores'].append(score)
                if score < 0.2:
                    logging.info('Low score: {} Score: {} Ground truth: {} File: {} Sample ID: {} Messages: {}'.format(model_output.replace('\n', ' '), score, sample.ground_truth, file, sample_id, messages[1:]))

            state['processed_file_ids'].add(file_sample_id)
        
        # save state
        with open(state_file, 'w') as f:
            state_copy = state.copy()
            state_copy['processed_file_ids'] = list(state_copy['processed_file_ids'])
            json.dump(state_copy, f, indent=4)


    if dialog_test_enabled:
        logging.info('Dialog score: {:.4f}'.format(np.mean(state['rougel_scores'])))
        with open(output_results_json, 'w') as f:
            json.dump({
                'dialog_score': np.mean(state['rougel_scores'])
            }, f, indent=4)

    if api_test_enabled:
        logging.info('Total API calls: {}'.format(state['total_api_calls']))
        logging.info('Correct API calls: {}'.format(state['correct_api_calls']))
        logging.info('Accuracy: {:.4f}'.format(state['correct_api_calls'] / state['total_api_calls']))
        with open(output_results_json, 'w') as f:
            json.dump({
                'total_api_calls': state['total_api_calls'],
                'correct_api_calls': state['correct_api_calls'],
                'accuracy': state['correct_api_calls'] / state['total_api_calls']
            }, f, indent=4)
