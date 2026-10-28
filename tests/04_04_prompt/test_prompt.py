"""Checks that the project from the prompt lesson is in place.

Our checks for the prompt lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Two checks that pinned the chat's instructions moved to the memory lesson
(04-05), which adds a paragraph of guidance after the prompt. One that pinned
where the prompt sits among the capabilities moved to the seam lesson
(04-06), where the prompt declares its place.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import shutil
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
PROMPT_FILE: Path = PROJECT_ROOT / "src" / "pydantic_tutor" / "prompts" / "chat.md"
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# The chat's prompt since the console-chat lesson, word for word.
PROMPT = "You are a helpful assistant. Answer briefly."
WORKSPACE_NAME = "Your workspace holds the Python project tinyjobs."


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder with a fresh `work/tinyjobs`, made by the script."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    made = run_script(name="fresh_copy.py", folder=tmp_path)
    assert made.returncode == 0, made.stderr
    return tmp_path


def run_script(*, name: str, folder: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
        env={
            **env,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
        },
    )


class Recorder:
    """A model that records the instructions of each request, then answers."""

    def __init__(self) -> None:
        self.instructions: list[str | None] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="Done.")])

    def model(self) -> FunctionModel:
        return FunctionModel(function=self.reply)


# Point 1: the prompt is a file, supplied by a capability.


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
    from pydantic_tutor.capabilities import prompt_file

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
    from pydantic_tutor.capabilities import prompt_file

    with pytest.raises(FileNotFoundError):
        prompt_file.PromptFile(path=tmp_path / "missing.md")
    empty = tmp_path / "empty.md"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prompt_file.PromptFile(path=empty)


# Point 2: the order of instruction text from several places.


def test_text_from_a_capability_comes_before_text_from_a_function(
    tmp_path: Path,
) -> None:
    # The rule the lesson teaches, on a second agent: the capability's text
    # is sent before the constructor's function, though listed after it.
    from pydantic_tutor.capabilities import prompt_file

    path = tmp_path / "prompt.md"
    path.write_text("From the file.", encoding="utf-8")

    def computed(ctx: RunContext[None]) -> str:
        return "From a function."

    recorder = Recorder()
    second = Agent(
        instructions=computed,
        capabilities=[prompt_file.PromptFile[None](path=path)],
    )
    second.run_sync(user_prompt="Hi.", model=recorder.model())
    assert recorder.instructions == ["From the file.\n\nFrom a function."]


def test_instruction_order_prints_static_then_dynamic(project: Path) -> None:
    result = run_script(name="instruction_order.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        " static: From the constructor, text.",
        f" static: {PROMPT}",
        " static: From the run, text.",
        f" static: {WORKSPACE_NAME}",
        "dynamic: From the constructor, a function.",
    ]
