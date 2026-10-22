"""Checks that the project from the lesson where a conversation survives a restart is in place.

Our checks for the restart lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model, and every line "typed" at the
keyboard is scripted too.

None of them touches your own `work/` folder or `pydantic-tutor.db`: each
uses a folder of its own.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import builtins
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest
from pydantic_ai import Agent, Conversation, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/05_04_restart/test_restart.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_repos" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(*, name: str, cwd: Path, stdin: str = "") -> str:
    """Run one of the project's scripts from `cwd`, `stdin` piped in; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        input=stdin,
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
    """A stand-in project folder: the sample repo and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_repos" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "restart_session.py"):
        # Before the lesson there is no restart_session.py: the test that
        # runs it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    run_script(name="scripts/fresh_copy.py", cwd=root)
    return root


def said(messages: list[ModelMessage]) -> list[str]:
    """Every prompt typed so far, in order."""
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A scripted model: says what it has been told so far."""
    return ModelResponse(parts=[TextPart(content=f"I can see: {said(messages)}")])


def conversation_after(*, prompt: str) -> Conversation:
    """The conversation one run on `echo` leaves."""
    return (
        Agent(model=FunctionModel(function=echo))
        .run_sync(user_prompt=prompt)
        .conversation
    )


def rows(*, path: Path) -> int:
    with closing(sqlite3.connect(database=path)) as connection:
        count: int = connection.execute(
            "SELECT count(*) FROM chat_conversations"
        ).fetchone()[0]
    return count


class Keyboard:
    """Scripted lines in place of `input`; once they run out, input has ended.

    A line may be a function: it is called when that line is asked for, and
    returns the line, so something else can happen first.
    """

    def __init__(self, *, lines: list[str | Callable[[], str]]) -> None:
        self.lines = list(lines)

    def __call__(self, prompt: object = "") -> str:
        if not self.lines:
            raise EOFError
        line = self.lines.pop(0)
        return line if isinstance(line, str) else line()


def open_chat(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lines: list[str | Callable[[], str]],
    function: Callable[[list[ModelMessage], AgentInfo], ModelResponse] = echo,
) -> None:
    """Open the student's chat on a scripted model, saving to tmp_path/c.db."""
    # The store's module is imported as a name of the package: before the
    # lesson it does not exist, and Ruff would sort a `from` import of it
    # apart from the rest of this block (I001).
    from pydantic_tutor import conversations
    from pydantic_tutor.agent import agent
    from pydantic_tutor.chat import chat
    from pydantic_tutor.composition import App
    from pydantic_tutor.repo_tools import RepoDeps

    monkeypatch.setattr("builtins.input", Keyboard(lines=lines))
    chat(
        app=App(
            agent=agent,
            model=FunctionModel(function=function, model_name="stand-in"),
            deps=RepoDeps(root=TINYJOBS_ROOT),
            store=conversations.ConversationStore(path=tmp_path / "c.db"),
        )
    )


def test_a_new_process_carries_on_the_saved_conversation(project: Path) -> None:
    # The lesson's point: the chat is closed, a new process opens it, and
    # the model is sent the first process's turn, with the same id and
    # the usage counted on.
    first = run_script(
        name="scripts/restart_session.py", cwd=project, stdin="My name is Ada.\n/exit\n"
    )
    assert "Turn 1. You've said: My name is Ada.\n" in first
    assert "Carrying on" not in first
    second = run_script(
        name="scripts/restart_session.py", cwd=project, stdin="What's my name?\n/exit\n"
    )
    assert "Carrying on conversation " in second
    assert ": 2 messages, usage.requests=1. Type /new to start another.\n" in second
    assert "Turn 2. You've said: My name is Ada. | What's my name?\n" in second
    assert (project / "work" / "conversations.db").is_file()


def test_the_store_saves_the_whole_conversation_and_a_new_store_reads_it(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.conversations import ConversationStore

    conversation = conversation_after(prompt="hello")
    revision = ConversationStore(path=tmp_path / "c.db").save(
        conversation=conversation, revision=0
    )
    assert revision == 1
    saved = ConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    assert saved.revision == 1
    loaded = saved.conversation
    assert loaded.conversation_id == conversation.conversation_id
    assert loaded.usage.requests == 1
    assert loaded.messages == conversation.messages


def test_nothing_is_made_until_first_use_and_an_empty_store_has_none(
    tmp_path: Path,
) -> None:
    # Building the store opens nothing: the app builds one at startup,
    # and earlier lessons' tests build the app.
    from pydantic_tutor.conversations import ConversationStore

    store = ConversationStore(path=tmp_path / "c.db")
    assert not (tmp_path / "c.db").exists()
    assert store.latest() is None


def test_a_save_replaces_the_earlier_one_and_latest_is_the_last_saved(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.conversations import ConversationStore

    store = ConversationStore(path=tmp_path / "c.db")
    a = conversation_after(prompt="a")
    b = conversation_after(prompt="b")
    store.save(conversation=a, revision=0)
    store.save(conversation=b, revision=0)
    latest = store.latest()
    assert latest is not None
    assert latest.conversation.conversation_id == b.conversation_id
    assert store.save(conversation=a, revision=1) == 2
    latest = store.latest()
    assert latest is not None
    assert latest.conversation.conversation_id == a.conversation_id
    assert latest.revision == 2
    assert rows(path=tmp_path / "c.db") == 2


def test_a_stale_save_is_refused_and_the_file_keeps_the_first(
    tmp_path: Path,
) -> None:
    # Two readers of one saved conversation: the second to save would
    # overwrite the first's turn, so it is refused instead.
    from pydantic_tutor.conversations import (
        ConversationConflictError,
        ConversationStore,
    )

    store = ConversationStore(path=tmp_path / "c.db")
    store.save(conversation=conversation_after(prompt="one"), revision=0)
    first = store.latest()
    second = store.latest()
    assert first is not None
    assert second is not None
    agent = Agent(model=FunctionModel(function=echo))
    one = agent.run_sync(user_prompt="from first", conversation=first.conversation)
    two = agent.run_sync(user_prompt="from second", conversation=second.conversation)
    store.save(conversation=one.conversation, revision=first.revision)
    with pytest.raises(ConversationConflictError):
        store.save(conversation=two.conversation, revision=second.revision)
    with pytest.raises(ConversationConflictError):
        # A conversation never saved cannot be saved as new twice either.
        store.save(conversation=one.conversation, revision=0)
    latest = store.latest()
    assert latest is not None
    assert said(latest.conversation.messages) == ["one", "from first"]
    assert latest.revision == 2


def test_the_chat_saves_each_turn_and_new_starts_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pydantic_tutor.conversations import ConversationStore

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one", "two"])
    saved = ConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    first = saved.conversation
    assert said(first.messages) == ["one", "two"]
    open_chat(
        monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["three", "/new", "four"]
    )
    out = capsys.readouterr().out
    assert "A new conversation starts with your next line.\n" in out
    latest = ConversationStore(path=tmp_path / "c.db").latest()
    assert latest is not None
    assert said(latest.conversation.messages) == ["four"]
    assert latest.conversation.conversation_id != first.conversation_id
    assert rows(path=tmp_path / "c.db") == 2


def test_two_chats_on_one_conversation_lose_no_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two chats open on the same saved conversation. The second saves a
    # turn while the first is waiting for a line; the first's next turn
    # is refused, and says so, instead of overwriting the second's.
    from pydantic_tutor.conversations import ConversationStore

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one"])
    capsys.readouterr()

    def second_chat_then_a_line() -> str:
        # Asked for by the first chat, which has read the conversation:
        # the second chat opens, saves a turn and closes; then the first
        # gets its line.
        first_keyboard = builtins.input
        open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["from second"])
        monkeypatch.setattr("builtins.input", first_keyboard)
        return "from first"

    open_chat(
        monkeypatch=monkeypatch, tmp_path=tmp_path, lines=[second_chat_then_a_line]
    )
    out = capsys.readouterr().out
    assert "This turn was not kept: " in out
    assert "I can see: ['one', 'from first']" not in out
    latest = ConversationStore(path=tmp_path / "c.db").latest()
    assert latest is not None
    assert said(latest.conversation.messages) == ["one", "from second"]


def test_a_turn_that_fails_is_not_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # What the chat carries and what is saved stay the same: the failed
    # turn is in neither.
    from pydantic_tutor.conversations import ConversationStore

    def fails_on_two(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if said(messages)[-1] == "two":
            raise UnexpectedModelBehavior("the model sent nonsense")
        return echo(messages, info)

    open_chat(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        lines=["one", "two"],
        function=fails_on_two,
    )
    saved = ConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    assert said(saved.conversation.messages) == ["one"]


def test_the_app_saves_to_the_file_the_settings_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    assert Settings().database == Path("pydantic-tutor.db")
    monkeypatch.setenv("PYDANTIC_TUTOR_DATABASE", str(tmp_path / "x.db"))
    app = build_app(settings=Settings(chat_model="test", repo_root=TINYJOBS_ROOT))
    assert app.store is not None
    assert app.store.path == tmp_path / "x.db"
    assert not (tmp_path / "x.db").exists()  # made on first use, by the chat


def test_git_ignores_the_saved_conversations() -> None:
    lines = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/pydantic-tutor.db" in lines
