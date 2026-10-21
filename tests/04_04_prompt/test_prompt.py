"""Checks that the project from the prompt lesson is in place.

Our checks for the prompt lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

One test was moved by the memory lesson (04-05), which adds the memory's
guidance after the prompt: what the model is sent is now proved in
`tests/04_05_memory/test_memory.py`, by
`test_the_model_is_sent_the_prompt_then_the_memory_then_the_repository`.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext, models
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_04_prompt/test_prompt.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
PROMPT_FILE: Path = (
    PROJECT_ROOT / "src" / "pydantic_tutor" / "prompts" / "repo_assistant.md"
)
# The agent's prompt since the agent lesson, word for word.
PROMPT = "You answer questions about a Python code repository. Answer briefly."


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> subprocess.CompletedProcess[str]:
    """Run one of the project's scripts, by its path from the project root."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result


class Recorder:
    """A model that records the instructions of each request, then answers."""

    def __init__(self) -> None:
        self.instructions: list[str | None] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="Done.")])


def test_the_prompt_is_a_file_in_the_package() -> None:
    assert PROMPT_FILE.read_text(encoding="utf-8").strip() == PROMPT


def test_no_prompt_text_is_left_in_python() -> None:
    # The lesson's point: the prompt lives in the file, not in code.
    sources = (PROJECT_ROOT / "src" / "pydantic_tutor").rglob("*.py")
    holding = [p.name for p in sources if PROMPT in p.read_text(encoding="utf-8")]
    assert holding == []


def test_prompt_file_is_a_capability_that_reads_its_file_when_made(
    tmp_path: Path,
) -> None:
    from pydantic_tutor import prompt_file

    path = tmp_path / "prompt.md"
    path.write_text("\nBe kind.\n\n", encoding="utf-8")
    cap: prompt_file.PromptFile[None] = prompt_file.PromptFile(path=path)
    assert isinstance(cap, AbstractCapability)
    # The text, without the space around it, as text: not a function.
    assert cap.get_instructions() == "Be kind."
    # Read once, when made: a later edit is not seen by this capability.
    path.write_text("Be brief.", encoding="utf-8")
    assert cap.get_instructions() == "Be kind."


def test_a_missing_or_empty_prompt_file_fails_when_made(tmp_path: Path) -> None:
    from pydantic_tutor import prompt_file

    with pytest.raises(FileNotFoundError):
        prompt_file.PromptFile(path=tmp_path / "missing.md")
    empty = tmp_path / "empty.md"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prompt_file.PromptFile(path=empty)


def test_the_agent_takes_its_prompt_from_the_file_first() -> None:
    from pydantic_tutor import prompt_file
    from pydantic_tutor.agent import PROMPT_PATH, agent

    assert PROMPT_PATH.resolve() == PROMPT_FILE.resolve()
    capabilities = agent.root_capability.capabilities
    prompts = [c for c in capabilities if isinstance(c, prompt_file.PromptFile)]
    assert len(prompts) == 1
    assert prompts[0].path == PROMPT_PATH
    # First of those the agent was given. The two the library adds of its
    # own, ToolSearch and PendingMessageDrainCapability, sit ahead of it.
    assert [type(c).__name__ for c in capabilities[:3]] == [
        "ToolSearch",
        "PendingMessageDrainCapability",
        "PromptFile",
    ]


def test_text_from_a_capability_comes_before_text_from_a_function(
    tmp_path: Path,
) -> None:
    # The rule the lesson teaches, on a second agent: the capability's text
    # is sent before the constructor's function, though listed after it.
    from pydantic_tutor import prompt_file

    path = tmp_path / "prompt.md"
    path.write_text("From the file.", encoding="utf-8")

    def computed(ctx: RunContext[None]) -> str:
        return "From a function."

    model = Recorder()
    second = Agent(
        instructions=computed,
        capabilities=[prompt_file.PromptFile[None](path=path)],
    )
    second.run_sync(user_prompt="Hi.", model=FunctionModel(function=model.reply))
    assert model.instructions == ["From the file.\n\nFrom a function."]


def test_instruction_order_prints_static_then_dynamic() -> None:
    lines = run_script("scripts/instruction_order.py").stdout.splitlines()
    assert lines == [
        " static: From the constructor, text.",
        f" static: {PROMPT}",
        " static: From the run, text.",
        "dynamic: From the constructor, a function.",
        "dynamic: From a capability, a function.",
        "dynamic: The repository is tinyjobs.",
    ]
