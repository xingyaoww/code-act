from .openai_lm_agent import OpenAILMAgent
import openai
import openai.error
import logging
import os
import traceback
from mint.datatypes import Action
import backoff

LOGGER = logging.getLogger("MINT")

MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MiniMaxLMAgent(OpenAILMAgent):
    """LLM agent using MiniMax's OpenAI-compatible API.

    MiniMax provides large language models (M2.7, M2.5) accessible via
    an OpenAI-compatible chat completions endpoint.

    Config:
        model_name: MiniMax model name (e.g. "MiniMax-M2.7")
        chat_mode: Must be True (MiniMax only supports chat completions)
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Sampling temperature, clamped to [0.01, 1.0] (default: 0)
    """

    def __init__(self, config):
        super().__init__(config)
        self.api_key = os.environ.get("MINIMAX_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required. "
                "Get your API key at https://platform.minimaxi.com/"
            )
        self.api_base = MINIMAX_API_BASE

    def _clamp_temperature(self, temperature):
        """Clamp temperature to MiniMax's accepted range [0.01, 1.0]."""
        if temperature <= 0:
            return 0.01
        return min(temperature, 1.0)

    @backoff.on_exception(
        backoff.fibo,
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )
    def call_lm(self, messages):
        temperature = self._clamp_temperature(
            self.config.get("temperature", 0)
        )
        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=temperature,
            stop=self.stop_words,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        # Convert OpenAIObject to plain dict for token counting
        usage_obj = response["usage"]
        usage = {}
        for k in usage_obj:
            try:
                usage[k] = int(usage_obj[k])
            except (TypeError, ValueError):
                pass
        return response.choices[0].message["content"], usage

    def act(self, state):
        messages = state.history
        try:
            lm_output, token_usage = self.call_lm(messages)
            for usage_type, count in token_usage.items():
                state.token_counter[usage_type] += count
            action = self.lm_output_to_action(lm_output)
            return action
        except openai.error.InvalidRequestError:
            tb = traceback.format_exc()
            return Action(f"", False, error=f"InvalidRequestError\n{tb}")
