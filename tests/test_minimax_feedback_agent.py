"""Unit tests for MiniMaxFeedbackAgent."""
import os
import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict

from mint.agents.minimax_feedback_agent import MiniMaxFeedbackAgent
from mint.datatypes import Action, State


@pytest.fixture
def feedback_config():
    return {
        "model_name": "MiniMax-M2.7",
        "chat_mode": True,
        "max_tokens": 1024,
        "temperature": 0.0,
    }


@pytest.fixture
def agent(feedback_config):
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
        return MiniMaxFeedbackAgent(feedback_config)


@pytest.fixture
def mock_state():
    state = State(
        history=[
            {
                "role": "user",
                "content": "System instructions\n---\nExamples\n---\nTask: Solve 2+2",
            },
            {"role": "assistant", "content": "Let me think about this."},
        ]
    )
    return state


class TestMiniMaxFeedbackAgentInit:
    def test_init_sets_feedback_stop_words(self, agent):
        assert "\nObservation:" in agent.stop_words
        assert "\nTask:" in agent.stop_words
        assert "\nAssistant:" in agent.stop_words

    def test_init_has_feedback_prompt(self, agent):
        assert agent.feedback_prompt is not None

    def test_inherits_minimax_api_config(self, agent):
        assert agent.api_base == "https://api.minimax.io/v1"
        assert agent.api_key == "test-key-123"


class TestFeedbackLMOutputToAction:
    def test_textual_feedback(self, agent):
        action = agent.lm_output_to_action(
            "The approach is correct but could be improved.", "textual"
        )
        assert isinstance(action, Action)
        assert action.value == "The approach is correct but could be improved."
        assert action.use_tool is False

    def test_binary_feedback_good(self, agent):
        action = agent.lm_output_to_action(
            "This is GOOD. The solution is correct.", "binary"
        )
        assert action.value == "This is GOOD."

    def test_binary_feedback_bad(self, agent):
        action = agent.lm_output_to_action(
            "This is BAD. The approach is wrong.", "binary"
        )
        assert action.value == "This is BAD."

    def test_binary_feedback_invalid(self, agent):
        with pytest.raises(ValueError, match="Cannot find GOOD or BAD in feedback"):
            agent.lm_output_to_action(
                "This is NEUTRAL. I'm not sure.", "binary"
            )


class TestFeedbackAct:
    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_textual_feedback(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message={"content": "The solution looks correct."}
            )
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }[key]
        mock_create.return_value = mock_response

        action = agent.act(
            state=mock_state,
            observation="Result: 4",
            form="textual",
            gt="4",
            task_in_context_example="Task: Example task",
            tool_desc="Python tool",
        )

        assert isinstance(action, Action)
        assert action.value == "The solution looks correct."
        assert action.use_tool is False
        assert mock_state.token_counter["feedback_prompt_tokens"] == 50

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_binary_feedback(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "This is GOOD. Well done."})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }[key]
        mock_create.return_value = mock_response

        action = agent.act(
            state=mock_state,
            observation="Result: 4",
            form="binary",
            gt="4",
            task_in_context_example="Task: Example task",
            tool_desc="Python tool",
        )

        assert action.value == "This is GOOD."

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_without_gt(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "Feedback without GT."})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
        }[key]
        mock_create.return_value = mock_response

        action = agent.act(
            state=mock_state,
            observation="Result: unknown",
            form="textual",
            gt=None,
            task_in_context_example="Task: Example task",
            tool_desc="Python tool",
        )

        assert action.value == "Feedback without GT."

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_invalid_request_error(self, mock_create, agent, mock_state):
        import openai.error
        mock_create.side_effect = openai.error.InvalidRequestError(
            "context length exceeded", ""
        )

        action = agent.act(
            state=mock_state,
            observation="Result: 4",
            form="textual",
            gt="4",
            task_in_context_example="Task: Example",
            tool_desc="Python tool",
        )

        assert action.error is not None
        assert "InvalidRequestError" in action.error

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_unknown_error(self, mock_create, agent, mock_state):
        mock_create.side_effect = RuntimeError("Something broke")

        action = agent.act(
            state=mock_state,
            observation="Result: 4",
            form="textual",
            gt="4",
            task_in_context_example="Task: Example",
            tool_desc="Python tool",
        )

        assert action.error is not None
        assert "Unknown error" in action.error

    @patch("mint.agents.minimax_lm_agent.openai.ChatCompletion.create")
    def test_act_tracks_feedback_tokens(self, mock_create, agent, mock_state):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message={"content": "Good work."})
        ]
        mock_response.__getitem__ = lambda self, key: {
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        }[key]
        mock_create.return_value = mock_response

        agent.act(
            state=mock_state,
            observation="Result: 4",
            form="textual",
            gt="4",
            task_in_context_example="Task: Example",
            tool_desc="Python tool",
        )

        assert mock_state.token_counter["feedback_prompt_tokens"] == 100
        assert mock_state.token_counter["feedback_completion_tokens"] == 20
