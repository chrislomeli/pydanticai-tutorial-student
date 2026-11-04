"""Checks that the project from the dependencies lesson is in place.

Our checks for the dependencies lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/01_04_deps/test_deps.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def ask_for_coffee(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that asks for the price of coffee, then answers with it."""
    for part in messages[-1].parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=str(part.content))])
    return ModelResponse(
        parts=[ToolCallPart(tool_name="price_of", args={"product": "coffee"})]
    )


def test_each_run_reads_its_own_deps() -> None:
    from pydantic_tutor.shop import Shop, shop_agent

    model = FunctionModel(function=ask_for_coffee)
    cheap = Shop(name="A", prices_in_cents={"coffee": 1})
    dear = Shop(name="B", prices_in_cents={"coffee": 2})

    assert shop_agent.run_sync(user_prompt="?", model=model, deps=cheap).output == "1"
    assert shop_agent.run_sync(user_prompt="?", model=model, deps=dear).output == "2"


def test_the_model_sees_the_instructions_text_not_the_deps() -> None:
    from pydantic_tutor.shop import Shop, shop_agent

    sent: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    shop = Shop(name="Corner Cafe", prices_in_cents={"coffee": 350})
    shop_agent.run_sync(user_prompt="?", model=FunctionModel(function=reply), deps=shop)

    assert sent[0].instructions is not None
    assert "Corner Cafe" in sent[0].instructions
    (price_of,) = sent[0].function_tools
    assert list(price_of.parameters_json_schema["properties"]) == ["product"]


def test_two_shops_prints_a_price_for_each() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/two_shops.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "output: Coffee is 350 cents." in result.stdout
    assert "output: Coffee is 500 cents." in result.stdout
