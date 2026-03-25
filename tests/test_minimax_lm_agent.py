"""Unit tests for MiniMaxLMAgent."""
import os
import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict

from mint.agents.minimax_lm_agent import MiniMaxLMAgent, MINIMAX_API_BASE
from mint.datatypes import Action, State


@pytest.fixture
def base_config():
    return {
        "model_name": "MiniMax-M2.7",
        "chat_mode": True,
        "max_tokens": 512,
        "temperature": 0.0,
    }


@pytest.fixture
def agent(base_config):
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
        return MiniMaxLMAgent(base_config)


@pytest.fixture
def mock_state():
    state = State(
        history=[
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    )
    return state


class TestMiniMaxLMAgentInit:
    def test_init_with_api_key(self, base_config):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent = MiniMaxLMAgent(base_config)
            assert agent.api_key == "test-key"
            assert agent.api_base == MINIMAX_API_BASE
            assert agent.config["model_name"] == "MiniMax-M2.7"

    def test_init_missing_api_key(self, base_config):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                MiniMaxLMAgent(base_config)

    def test_init_missing_model_name(self):
        config = {"chat_mode": True}
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with pytest.raises(AssertionError):
                MiniMaxLMAgent(config)

    def test_api_base_url(self, agent):
        assert agent.api_base == "https://api.minimax.io/v1"

    def test_stop_words_inherited(self, agent):
        assert "\nObservation:" in agent.stop_words
        assert "\nExpert feedback:" in agent.stop_words
        assert "\nTask:" in agent.stop_words
        assert "\n---" in agent.stop_words


class TestTemperatureClamping:
    def test_clamp_zero_temperature(self, agent):
        assert agent._clamp_temperature(0) == 0.01

    def test_clamp_negative_temperature(self, agent):
        assert agent._clamp_temperature(-1.0) == 0.01

    def test_clamp_valid_temperature(self, agent):
        assert agent._clamp_temperature(0.5) == 0.5

    def test_clamp_max_temperature(self, agent):
        assert agent._clamp_temperature(1.0) == 1.0

    def test_clamp_over_max_temperature(self, agent):
        assert agent._clamp_temperature(2.0) == 1.0

    def test_clamp_small_positive(self, agent):
        assert agent._clamp_temperature(0.01) == 0.01

    def test_clamp_boundary_value(self, agent):
        assert agent._clamp_temperature(0.001) == 0.001


class TestCallLM:
    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_call_lm_success(self, mock_create, agent):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "The answer is 4."})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }[key]
        mock_create.return_value = mock_response

        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        content, usage = agent.call_lm(messages)

        assert content == "The answer is 4."
        mock_create.assert_called_once_with(
            model="MiniMax-M2.7",
            messages=messages,
            max_tokens=512,
            temperature=0.01,  # clamped from 0.0
            stop=agent.stop_words,
            api_base=MINIMAX_API_BASE,
            api_key="test-key-123",
        )

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_call_lm_uses_config_temperature(self, mock_create, base_config):
        base_config["temperature"] = 0.7
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent = MiniMaxLMAgent(base_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"content": "Response"})]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }[key]
        mock_create.return_value = mock_response

        agent.call_lm([{"role": "user", "content": "Hi"}])
        call_args = mock_create.call_args
        assert call_args.kwargs["temperature"] == 0.7

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_call_lm_custom_max_tokens(self, mock_create, base_config):
        base_config["max_tokens"] = 1024
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent = MiniMaxLMAgent(base_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"content": "Response"})]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }[key]
        mock_create.return_value = mock_response

        agent.call_lm([{"role": "user", "content": "Hi"}])
        call_args = mock_create.call_args
        assert call_args.kwargs["max_tokens"] == 1024


class TestAct:
    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_success(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "The answer is 4."})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }[key]
        mock_create.return_value = mock_response

        action = agent.act(mock_state)
        assert isinstance(action, Action)
        assert action.value == "The answer is 4."
        assert action.use_tool is True  # no <solution> tag
        assert mock_state.token_counter["prompt_tokens"] == 10
        assert mock_state.token_counter["completion_tokens"] == 5

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_with_solution_tag(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "<solution>4</solution>"})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }[key]
        mock_create.return_value = mock_response

        action = agent.act(mock_state)
        assert action.use_tool is False  # has <solution> tag

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_invalid_request_error(self, mock_create, agent, mock_state):
        import openai.error
        mock_create.side_effect = openai.error.InvalidRequestError(
            "context length exceeded", ""
        )
        action = agent.act(mock_state)
        assert action.error is not None
        assert "InvalidRequestError" in action.error

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_tracks_token_usage(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"content": "Response"})]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }[key]
        mock_create.return_value = mock_response

        agent.act(mock_state)
        assert mock_state.token_counter["prompt_tokens"] == 20
        assert mock_state.token_counter["completion_tokens"] == 10
        assert mock_state.token_counter["total_tokens"] == 30


class TestModelNames:
    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_m27_model(self, mock_create):
        config = {"model_name": "MiniMax-M2.7", "chat_mode": True}
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            agent = MiniMaxLMAgent(config)
        assert agent.config["model_name"] == "MiniMax-M2.7"

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_m27_highspeed_model(self, mock_create):
        config = {"model_name": "MiniMax-M2.7-highspeed", "chat_mode": True}
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key"}):
            agent = MiniMaxLMAgent(config)
        assert agent.config["model_name"] == "MiniMax-M2.7-highspeed"


class TestModuleExports:
    def test_import_from_agents(self):
        from mint.agents import MiniMaxLMAgent
        assert MiniMaxLMAgent is not None

    def test_import_from_agents_feedback(self):
        from mint.agents import MiniMaxFeedbackAgent
        assert MiniMaxFeedbackAgent is not None
