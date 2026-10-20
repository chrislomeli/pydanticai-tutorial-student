"""Checks that the project from the chat lesson is in place.

Our checks for the chat lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`. They fail until the
lesson is done. None of them needs an API key: `to_cli_sync()`'s own
interactive loop is for a person in a terminal, not pytest, and is checked
by the recipe run and at Accept instead.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def scripted_model() -> FunctionModel:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hi there")])

    return FunctionModel(function=reply)


def test_configure_tracing_is_a_no_op_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Goal 2, off: with PYDANTIC_TUTOR_TRACE unset, configure_tracing()
    # does nothing observable.
    from pydantic_tutor.__main__ import TRACE_VAR, configure_tracing

    monkeypatch.delenv(TRACE_VAR, raising=False)
    configure_tracing()
    assert capsys.readouterr().out == ""


def test_configure_tracing_prints_spans_for_a_scripted_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Goal 2, on: with the flag set, a scripted turn's spans print to this
    # terminal -- no key, no network call, nothing sent anywhere.
    from pydantic_tutor.__main__ import TRACE_VAR, configure_tracing
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    monkeypatch.setenv(TRACE_VAR, "1")
    configure_tracing()
    capsys.readouterr()  # discard the "tracing on" line configure_tracing printed

    agent.run_sync(
        user_prompt="hello",
        model=scripted_model(),
        deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
    )

    assert "repo_assistant run" in capsys.readouterr().out


def test_a_missing_key_fails_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Goal 1, before the key: opening the chat against a real provider
    # with no key set raises a UserError naming the variable, not a stack
    # trace to decode.
    from pydantic_ai.exceptions import UserError

    from pydantic_tutor.__main__ import DEFAULT_MODEL
    from pydantic_tutor.agent import agent
    from pydantic_tutor.repo_tools import RepoDeps

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(UserError) as excinfo:
        agent.run_sync(
            user_prompt="hello",
            model=DEFAULT_MODEL,
            deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
        )
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_the_entry_point_uses_the_project_s_one_agent() -> None:
    # The entry point is wired to the agent every other script already
    # uses, not a second one of its own; DEFAULT_MODEL names a real
    # provider, so the entry point defaults to talking to one, with a
    # scripted model substituted here to prove the wiring without a key.
    from pydantic_tutor import agent as agent_module
    from pydantic_tutor.__main__ import DEFAULT_MODEL
    from pydantic_tutor.repo_tools import RepoDeps

    assert DEFAULT_MODEL.startswith("openai:")

    result = agent_module.agent.run_sync(
        user_prompt="hello",
        model=scripted_model(),
        deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
    )
    assert result.output == "hi there"
