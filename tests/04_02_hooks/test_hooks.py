"""Checks that the project from the hooks lesson is in place.

Our checks for the hooks lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import RunContext, models
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

# This file is tests/04_02_hooks/test_hooks.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
LOGGER = "pydantic_tutor.capabilities.request_log"


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
    """A model that records what each request carries, calling list_modules first."""

    def __init__(self) -> None:
        self.tool_counts: list[int] = []
        self.message_counts: list[int] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tool_counts.append(len(info.function_tools))
        self.message_counts.append(len(messages))
        if len(self.tool_counts) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="list_modules", args={})]
            )
        return ModelResponse(parts=[TextPart(content="ok")])


def logged(*, caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages our request log wrote during the test."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


# Point 1: a hook that logs each request, as Hooks(...) and as a class.


def test_the_request_log_is_a_hooks_capability_last_on_the_chat(
    project: Path,
) -> None:
    from pydantic_tutor.capabilities import request_log as module
    from pydantic_tutor.chat.agent import build_chat_agent

    assert isinstance(module.request_log, Hooks)
    assert module.logger.name == LOGGER
    # Last, with web search on or off, so it sees each request as every
    # other capability has left it.
    for web_search in (False, True):
        agent = build_chat_agent(
            web_search=web_search, workspace_root=project / "work" / "tinyjobs"
        )
        assert agent.root_capability.capabilities[-1] is module.request_log


def test_each_request_is_logged_with_its_step_messages_and_tools(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder()
    chat = build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    )
    chat.run_sync(user_prompt="x", model=FunctionModel(function=recorder.reply))
    assert logged(caplog=caplog) == [
        "request step=1 messages=1 tools=14",
        "request step=2 messages=3 tools=14",
    ]
    # The hook only watches: the model got what the log describes.
    assert recorder.message_counts == [1, 3]
    assert recorder.tool_counts == [14, 14]


def test_the_log_counts_the_tools_actually_sent(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent

    # Hide every tool but ours for one run: the log must count what is
    # sent, not a number fixed when the agent was built.
    def only_ours(
        ctx: RunContext[None], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            tool
            for tool in tool_defs
            if tool.name in {"list_modules", "read_docstring"}
        ]

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder()
    chat = build_chat_agent(
        web_search=False, workspace_root=project / "work" / "tinyjobs"
    )
    chat.run_sync(
        user_prompt="x",
        model=FunctionModel(function=recorder.reply),
        capabilities=[Hooks(prepare_tools=only_ours)],
    )
    assert recorder.tool_counts == [2, 2]
    assert logged(caplog=caplog) == [
        "request step=1 messages=1 tools=2",
        "request step=2 messages=3 tools=2",
    ]


def test_watch_requests_prints_the_log(project: Path) -> None:
    result = run_script(name="watch_requests.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == [
        f"{LOGGER}: request step=1 messages=1 tools=14",
        f"{LOGGER}: request step=2 messages=3 tools=14",
    ]
    assert lines[2].startswith("output: ")
    assert lines[3].startswith("usage: ")


# Point 2: where a hook stands decides what it sees; a module only writes.


def test_nothing_is_logged_until_the_program_turns_it_on(project: Path) -> None:
    # Earlier script that runs the chat's agent and configures no logging:
    # its output is what it was before the lesson.
    result = run_script(name="workspace_reach.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert "request step=" not in result.stdout
    assert "request step=" not in result.stderr


def turn(*, step: int, logged: int, sent: int) -> list[str]:
    """The two lines one request prints: the log's count, then the model's."""
    return [
        f"{LOGGER}: request step={step} messages={logged} tools=2",
        f"  the model receives: messages={sent} tools=2",
    ]


def test_hook_forms_shows_why_the_position_matters(project: Path) -> None:
    result = run_script(name="hook_forms.py", folder=project)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "the log, as a class, before a hook that trims the history:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        # It logs the request as it was before the trimmer shortened it.
        *turn(step=3, logged=5, sent=3),
        "the log, as a class, after it:",
        *turn(step=1, logged=1, sent=1),
        *turn(step=2, logged=3, sent=3),
        *turn(step=3, logged=3, sent=3),
    ]
