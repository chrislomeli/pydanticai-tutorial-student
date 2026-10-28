"""Checks that the project from the web-search lesson is in place.

Our checks for the web-search lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them calls a real model.
"""

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.native_tools import WebSearchTool


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def test_the_chat_agent_offers_the_model_web_search() -> None:
    from pydantic_tutor.chat_agent import chat_agent

    sent: list[ModelRequestParameters] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info.model_request_parameters)
        return ModelResponse(parts=[TextPart(content="ok")])

    chat_agent.run_sync(user_prompt="hi", model=FunctionModel(function=reply))

    assert any(isinstance(tool, WebSearchTool) for tool in sent[0].native_tools)
