"""Checks that the project from the tools lesson is in place.

Our checks for the tools lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`. They fail until the
lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry, models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/01_03_function_tools/test_function_tools.py: two
# folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        [sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_a_tool_checks_the_name_the_model_wrote() -> None:
    from pydantic_tutor.prices import price_of

    assert price_of(product="coffee") == 700
    with pytest.raises(ModelRetry, match="list_products"):
        price_of(product="apples")


def test_the_model_is_told_each_tools_name_description_and_schema() -> None:
    from pydantic_tutor.prices import price_agent

    sent: list[ToolDefinition] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.extend(info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    price_agent.run_sync(user_prompt="anything", model=FunctionModel(function=reply))

    tools = {tool.name: tool for tool in sent}
    assert set(tools) == {"list_products", "price_of"}
    price_of = tools["price_of"]
    assert price_of.description == "Look up the price of one product, in cents."
    assert price_of.parameters_json_schema["required"] == ["product"]


def test_a_wrong_name_is_retried_and_the_run_makes_three_requests() -> None:
    printed = run_script("scripts/tool_loop.py")

    assert "request ['retry-prompt']" in printed
    assert "output: An apple costs 50 cents." in printed
    assert printed.strip().splitlines()[-1] == "requests: 3"
