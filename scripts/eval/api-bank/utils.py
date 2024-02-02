import openai
import backoff
import logging
import requests
import os
import json
import google.generativeai as genai

logger = logging.getLogger("backoff")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class GeminiWrapper:

    ROLE_MAPPING = {
        "system": "system",
        "assistant": "model",
        "user": "user"
    }
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        assert "gemini" in self.model_name
        self.model = genai.GenerativeModel(self.model_name)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        logger=logger,
    )
    def call(self, messages, **kwargs):
        # https://github.com/neulab/gemini-benchmark/blob/6b3b8f18c3fbaa6df947f00fc49b87802c5e063e/benchmarking/Code/run_code.py#L17
        extra_kwargs = {}
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        extra_kwargs = {
            "safety_settings": safety_settings,
        }
        # Convert message format
        # messages = [
        #     {'role':'user',
        #     'parts': ["Briefly explain how a computer works to a young child."]}
        # ]
        messages = [
            {
                "role": self.ROLE_MAPPING[message["role"]],
                "parts": [message["content"]]
            }
            for message in messages
        ]
        # merge the system message with the first user message
        # since Gemini doesn't support system message
        if messages[0]["role"] == "system":
            if len(messages) > 1:
                assert messages[1]["role"] == "user"
                # merge the system message with the first user message
                messages[1]["parts"][0] = f"{messages[0]['parts'][0]}\n\n{messages[1]['parts'][0]}"
                messages = messages[1:]
            else:
                # if there is only one message, we just set the role to user
                messages[0]["role"] = "user"

        # if the last message is assistant, we need to add a user message (Gemini restriction)
        if messages[-1]["role"] == "model":
            messages.append({
                "role": "user",
                "parts": ["Continue."]
            })

        # concatenate any two consecutive user messages (required by Gemini)
        new_messages = []
        for message in messages:
            if message["role"] == "user" \
                and len(new_messages) > 0 and new_messages[-1]["role"] == "user":
                new_messages[-1]["parts"][0] = f"{new_messages[-1]['parts'][0]}\n\n{message['parts'][0]}"
            else:
                new_messages.append(message)
        messages = new_messages

        for message in messages:
            print(message["role"], message["parts"])

        response = self.model.generate_content(
            messages,
            **extra_kwargs
        )
        return response.text

class OpenAIChatWrapper:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
            
        ),
        logger=logger,
    )
    def call(self, messages, **kwargs):
        query = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
        }
        query.update(kwargs)
        try:
            response = openai.ChatCompletion.create(**query)
        except openai.error.APIError as e:
            if "maximum context length" in str(e):
                return None
            raise e
        return response


class DavinciWrapper:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
        logger=logger,
    )
    def call(self, messages, **kwargs):

        # messages to prompt
        prompt = ''
        for message in messages:
            prompt += message['role'] + ': ' + message['content'] + '\n'

        query = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 512,
        }
        query.update(kwargs)
        response = openai.Completion.create(**query)
        return response

class ClaudeWrapper:

    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
    }
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
    )
    def call(self, messages, **kwargs):
        # Prepend the prompt with the system message
        data = {
            "model": self.model_name,
            "prompt": "",
            "max_tokens_to_sample": 512,
            # "temperature": self.config.get("temperature", 0),
            # "stop_sequences": self.stop_words,
        }
        for message in messages:
            if message["role"] == "user" or message["role"] == "system":
                data["prompt"] += f"\n\nHuman: {message['content']}"
            else:
                data["prompt"] += f"\n\nAssistant: {message['content']}"
        data["prompt"] += "\n\nAssistant:"

        response = requests.post(self.url, headers=self.headers, json=data)

        if response.status_code == 200:
            pass
        else:
            logger.error(response.text)
            raise requests.exceptions.RequestException(
                "Request failed with status code:", response.status_code
            )

        return json.loads(response.text)["completion"]
