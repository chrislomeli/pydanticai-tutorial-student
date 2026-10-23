"""Checks that the project from the file tools lesson is in place.

Our checks for the file tools lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model.

None of them touches your own `work/` folder: each makes a working copy of
its own, with your `scripts/fresh_copy.py`, in a folder of its own.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

Five tests here were moved from earlier lessons, which this lesson's
changes break: four from the settings lesson (03-01) and one from the
models lesson (03-02). Each says so. Four of them prove an earlier
lesson's point, so they pass before this lesson too: the ones that build
the app, or open the chat.

One test was moved on by the shell lesson (03-05), which gives the agent
a shell beside its file tools: that the model is sent our two tools'
signatures beside the library's tools is now proved in
`tests/03_05_shell/test_shell.py`, by
`test_the_model_is_sent_our_signatures_beside_the_file_and_shell_tools`.

Two tests were moved on by the approval lesson (05-01), which holds every
write for a person's approval, and refuses it when no one answers: that
the file tools write in the working copy, and that the default write
protection is kept, are now proved with the write approved, in
`tests/05_01_approval/test_approval.py`, by
`test_an_approved_write_is_written_in_the_working_copy` and
`test_an_approved_write_still_keeps_the_default_protection`.
"""

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel

# This file is tests/03_04_files/test_files.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
FAKE_TOKEN = "tj-FAKE-0000-not-a-real-secret"
PREFIX = "PYDANTIC_TUTOR_"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(*, name: str, cwd: Path) -> str:
    """Run one of the lesson's scripts from `cwd`; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample codebase and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_code" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "file_tools.py"):
        shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    return root


@pytest.fixture
def working_copy(project: Path) -> Path:
    """A working copy, made by the student's `scripts/fresh_copy.py`."""
    run_script(name="scripts/fresh_copy.py", cwd=project)
    return project / "work" / "tinyjobs"


@dataclass(frozen=True)
class ToolResult:
    """What one file tool call gave the model."""

    refused: bool  # True: a retry prompt, asking the model to try again
    text: str
    # Every message of the run, to look for a secret anywhere in it.
    messages: list[ModelMessage]


def call(*, root: Path, tool_name: str, args: dict[str, object]) -> ToolResult:
    """Run the student's agent on `root`, with a model that makes one tool call."""
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart | RetryPromptPart):
                return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    result = agent.run_sync(
        user_prompt="x",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=root),
    )
    messages = result.all_messages()
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ToolResult(
                    refused=False, text=str(part.content), messages=messages
                )
            if isinstance(part, RetryPromptPart):
                return ToolResult(
                    refused=True, text=str(part.content), messages=messages
                )
    raise AssertionError("the tool was never called")


def mentions(*, messages: list[ModelMessage], text: str) -> bool:
    """Whether `text` appears in any part of any message, to or from the model."""
    return any(text in str(part) for message in messages for part in message.parts)


# Goal 1: the agent has file tools, in a working copy it may change.


def test_fresh_copy_makes_a_working_copy_with_a_fake_env(project: Path) -> None:
    printed = run_script(name="scripts/fresh_copy.py", cwd=project)
    working_copy = project / "work" / "tinyjobs"
    assert printed.strip() == (
        "work/tinyjobs: a fresh copy of sample_code/tinyjobs, with a fake .env"
    )
    assert (working_copy / "src" / "tinyjobs" / "jobs.py").read_text() == (
        TINYJOBS_ROOT / "src" / "tinyjobs" / "jobs.py"
    ).read_text()
    assert (working_copy / "tests" / "test_jobs.py").is_file()  # the tests come too
    assert FAKE_TOKEN in (working_copy / ".env").read_text()


def test_fresh_copy_replaces_an_earlier_copy(working_copy: Path, project: Path) -> None:
    (working_copy / "README.md").write_text("changed\n")
    (working_copy / "notes.txt").write_text("left by the agent\n")
    run_script(name="scripts/fresh_copy.py", cwd=project)
    assert (working_copy / "README.md").read_text() == (
        TINYJOBS_ROOT / "README.md"
    ).read_text()
    assert not (working_copy / "notes.txt").exists()


def test_the_working_copy_is_kept_out_of_git_and_the_tools() -> None:
    ignored = subprocess.run(
        args=[
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "work/tinyjobs/README.md",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert ignored.returncode == 0, "work/ is not in .gitignore"
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    tool = config["tool"]
    assert "work" in tool["ruff"]["extend-exclude"]
    assert "^work/" in tool["mypy"]["exclude"]
    assert "--ignore=work" in tool["pytest"]["addopts"]


@pytest.mark.parametrize(
    argnames="path", argvalues=["../outside.txt", "ABSOLUTE", "escape.txt"]
)
def test_the_file_tools_stay_in_the_folder(working_copy: Path, path: str) -> None:
    # A guard the stock file tools meet: `..`, an absolute path, and a link
    # out of the folder, all leading to the same file outside it.
    outside = working_copy.parent / "outside.txt"
    outside.write_text("outside the working copy\n")
    (working_copy / "escape.txt").symlink_to(target=outside)
    target = str(outside) if path == "ABSOLUTE" else path
    result = call(root=working_copy, tool_name="read_file", args={"path": target})
    assert result.refused
    assert "outside" in result.text
    assert not mentions(messages=result.messages, text="outside the working copy")


# Goal 2: the secret files cannot be read, as well as not written.


def test_the_env_file_cannot_be_read(working_copy: Path) -> None:
    result = call(root=working_copy, tool_name="read_file", args={"path": ".env"})
    assert result.refused
    assert "denied" in result.text
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


@pytest.mark.parametrize(
    argnames="path",
    argvalues=[
        "src/tinyjobs/.env",  # in a folder below the top one
        ".env.local",
        "deploy/runner.pem",
        "deploy/runner.key",
        "config/secrets.toml",
    ],
)
def test_other_secret_files_cannot_be_read(working_copy: Path, path: str) -> None:
    secret = working_copy / path
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text(f"TOKEN={FAKE_TOKEN}\n")
    result = call(root=working_copy, tool_name="read_file", args={"path": path})
    assert result.refused, f"{path} was read"
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_a_link_to_the_env_file_cannot_be_read(working_copy: Path) -> None:
    (working_copy / "settings.txt").symlink_to(target=".env")
    result = call(
        root=working_copy, tool_name="read_file", args={"path": "settings.txt"}
    )
    assert result.refused
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_a_search_does_not_reveal_a_secret(working_copy: Path) -> None:
    # `.env` is a hidden file, which a search skips anyway; a key file is not.
    (working_copy / "deploy").mkdir()
    (working_copy / "deploy" / "runner.pem").write_text(f"TOKEN={FAKE_TOKEN}\n")
    result = call(
        root=working_copy, tool_name="search_files", args={"pattern": "tj-FAKE"}
    )
    assert not mentions(messages=result.messages, text=FAKE_TOKEN)


def test_file_tools_prints_what_is_read_and_refused(
    working_copy: Path, project: Path
) -> None:
    printed = run_script(name="scripts/file_tools.py", cwd=project).splitlines()
    assert printed[:4] == [
        "read README.md:",
        "  [README.md | 26 lines | hash:d7795053b8e5]",
        "       1\t# tinyjobs",
        "  ... (25 more lines. Use offset=1 to continue reading.)",
    ]
    assert printed[4:8] == [
        "read .env:",
        "  Path '.env' is denied by pattern '**/.env'.",
        "write .env:",
        "  Path '.env' is protected (matches '**/.env').",
    ]
    assert printed[8] == "read ../../pyproject.toml:"
    assert "is outside the project root" in printed[9]
    assert FAKE_TOKEN not in "\n".join(printed)


# The app works on the working copy, and will not start without it.


def test_settings_default_to_the_working_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Moved from 03-01, where the default was the sample codebase itself. Run in
    # an empty folder, so that your own `.env`, if you have one, is not read.
    from pydantic_tutor.settings import Settings

    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.chat_model == "openai:gpt-5-mini"
    assert settings.trace is False
    assert settings.codebase_root == Path("work/tinyjobs")


def test_the_app_will_not_start_without_its_codebase(tmp_path: Path) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    missing = tmp_path / "nowhere"
    with pytest.raises(UserError) as excinfo:
        build_app(settings=Settings(chat_model="test", codebase_root=missing))
    message = str(excinfo.value)
    assert str(missing) in message
    assert "PYDANTIC_TUTOR_CODEBASE_ROOT" in message
    assert "scripts/fresh_copy.py" in message


def test_build_app_builds_the_objects_from_the_settings(tmp_path: Path) -> None:
    # Moved from 03-01: its codebase, "elsewhere", now has to exist.
    from pydantic_tutor import agent as agent_module
    from pydantic_tutor import composition
    from pydantic_tutor import settings as settings_module

    app = composition.build_app(
        settings=settings_module.Settings(chat_model="test", codebase_root=tmp_path)
    )
    assert app.agent is agent_module.agent  # the project's one agent, not a second
    assert isinstance(app.model, TestModel)  # an object, built from the string
    assert app.deps.root == tmp_path


def test_tracing_stays_off_without_the_setting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Moved from 03-01, which built the app on the default codebase.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    build_app(settings=Settings(chat_model="test", codebase_root=TINYJOBS_ROOT))
    assert capsys.readouterr().out == ""


def test_tracing_prints_spans_for_a_scripted_turn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Moved from 03-01, which built the app on the default codebase.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hi there")])

    app = build_app(
        settings=Settings(chat_model="test", trace=True, codebase_root=TINYJOBS_ROOT)
    )
    capsys.readouterr()  # discard the "tracing on" line
    app.agent.run_sync(
        user_prompt="hello", model=FunctionModel(function=reply), deps=app.deps
    )
    assert "assistant run" in capsys.readouterr().out


def test_the_entry_point_opens_the_chat_with_the_built_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Moved from 03-02, which ran it in an empty folder: the default
    # codebase, `work/tinyjobs`, now has to be there.
    import pydantic_tutor.__main__ as entry
    from pydantic_tutor import agent as agent_module

    monkeypatch.chdir(tmp_path)
    (tmp_path / "work" / "tinyjobs").mkdir(parents=True)
    Path(".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
    opened: dict[str, object] = {}

    def fake_to_cli_sync(**kwargs: object) -> None:
        opened.update(kwargs)

    monkeypatch.setattr(
        target=agent_module.agent, name="to_cli_sync", value=fake_to_cli_sync
    )
    entry.main()
    model = opened["model"]
    assert isinstance(model, OpenAIResponsesModel)
    assert model.client.api_key == "sk-from-dotenv"
    assert "OPENAI_API_KEY" not in os.environ  # read into Settings only
