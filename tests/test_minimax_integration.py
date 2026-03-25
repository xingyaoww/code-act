"""Integration tests for MiniMax agents.

These tests verify the MiniMax agents integrate correctly with the
code-act framework's agent factory and configuration system.
Requires MINIMAX_API_KEY to be set for live API tests.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import mint.agents as agents
from mint.agents.minimax_lm_agent import MiniMaxLMAgent
from mint.agents.minimax_feedback_agent import MiniMaxFeedbackAgent
from mint.configs.config_variables import EVALUATED_MODEL_LIST
from mint.datatypes import State


class TestAgentFactoryIntegration:
    """Test that MiniMax agents work with the dynamic agent instantiation pattern."""

    def test_minimax_agent_in_evaluated_model_list(self):
        minimax_entries = [
            e for e in EVALUATED_MODEL_LIST
            if e["agent_class"] == "MiniMaxLMAgent"
        ]
        assert len(minimax_entries) >= 2
        model_names = [e["config"]["model_name"] for e in minimax_entries]
        assert "MiniMax-M2.7" in model_names
        assert "MiniMax-M2.7-highspeed" in model_names

    def test_dynamic_instantiation_lm_agent(self):
        """Test the getattr(agents, agent_class)(config) pattern from main.py."""
        agent_config = {
            "agent_class": "MiniMaxLMAgent",
            "config": {
                "model_name": "MiniMax-M2.7",
                "chat_mode": True,
                "max_tokens": 512,
                "temperature": 0.0,
            },
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent_cls = getattr(agents, agent_config["agent_class"])
            agent = agent_cls(agent_config["config"])
            assert isinstance(agent, MiniMaxLMAgent)

    def test_dynamic_instantiation_feedback_agent(self):
        agent_config = {
            "agent_class": "MiniMaxFeedbackAgent",
            "config": {
                "model_name": "MiniMax-M2.7",
                "chat_mode": True,
                "max_tokens": 1024,
                "temperature": 0.0,
            },
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent_cls = getattr(agents, agent_config["agent_class"])
            agent = agent_cls(agent_config["config"])
            assert isinstance(agent, MiniMaxFeedbackAgent)

    def test_all_evaluated_models_instantiable(self):
        """Verify all MiniMax entries in EVALUATED_MODEL_LIST can be instantiated."""
        for entry in EVALUATED_MODEL_LIST:
            if entry["agent_class"] == "MiniMaxLMAgent":
                with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
                    agent = getattr(agents, entry["agent_class"])(entry["config"])
                    assert agent.api_base == "https://api.minimax.io/v1"


class TestEndToEndFlow:
    """Test the full agent flow: instantiate -> act -> get action."""

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_full_act_flow(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message={"content": "I'll solve this step by step.\n```python\nprint(2+2)\n```"}
            )
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        }[key]
        mock_create.return_value = mock_response

        config = {
            "model_name": "MiniMax-M2.7",
            "chat_mode": True,
            "max_tokens": 512,
            "temperature": 0.5,
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent = MiniMaxLMAgent(config)

        state = State(
            history=[
                {
                    "role": "user",
                    "content": "System prompt\n---\nExamples\n---\nTask: What is 2+2?",
                }
            ]
        )

        action = agent.act(state)

        # Verify action
        assert action.use_tool is True  # no <solution> tag
        assert "python" in action.value
        # Verify token tracking
        assert state.token_counter["total_tokens"] == 70
        # Verify API call parameters
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["api_base"] == "https://api.minimax.io/v1"
        assert call_kwargs["temperature"] == 0.5

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_multi_turn_conversation(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "<solution>4</solution>"})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90}
        }[key]
        mock_create.return_value = mock_response

        config = {"model_name": "MiniMax-M2.7", "chat_mode": True}
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            agent = MiniMaxLMAgent(config)

        state = State(
            history=[
                {"role": "user", "content": "System\n---\nEx\n---\nTask: 2+2"},
                {"role": "assistant", "content": "Let me compute this."},
                {"role": "user", "content": "Observation:\n4\nYou have 4 steps left."},
            ]
        )

        action = agent.act(state)
        assert action.use_tool is False  # has <solution>
        assert "<solution>" in action.value

        # Verify all 3 messages were passed
        call_kwargs = mock_create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set"
)
class TestLiveAPI:
    """Live API tests - only run when MINIMAX_API_KEY is available."""

    def test_live_chat_completion(self):
        config = {
            "model_name": "MiniMax-M2.7",
            "chat_mode": True,
            "max_tokens": 50,
            "temperature": 0.01,
        }
        agent = MiniMaxLMAgent(config)
        state = State(
            history=[{"role": "user", "content": "Reply with exactly: hello"}]
        )
        action = agent.act(state)
        assert action.error is None
        assert len(action.value) > 0

    def test_live_m27_highspeed(self):
        config = {
            "model_name": "MiniMax-M2.7-highspeed",
            "chat_mode": True,
            "max_tokens": 50,
            "temperature": 0.01,
        }
        agent = MiniMaxLMAgent(config)
        state = State(
            history=[{"role": "user", "content": "Reply with exactly: world"}]
        )
        action = agent.act(state)
        assert action.error is None
        assert len(action.value) > 0

    def test_live_token_tracking(self):
        config = {
            "model_name": "MiniMax-M2.7",
            "chat_mode": True,
            "max_tokens": 30,
            "temperature": 0.01,
        }
        agent = MiniMaxLMAgent(config)
        state = State(
            history=[{"role": "user", "content": "Say hi"}]
        )
        agent.act(state)
        assert state.token_counter["prompt_tokens"] > 0
        assert state.token_counter["completion_tokens"] > 0
