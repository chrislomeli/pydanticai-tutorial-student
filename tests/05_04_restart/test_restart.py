"""Checks that the project from the restart lesson is in place.

Our checks for the restart lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request: every
run is on a stand-in model, and the person at the keyboard is a list of
typed lines.

Each test runs in an empty folder of its own, so the project's own `.env`,
`work/` folder and `pydantic-tutor.db` are never touched. A test that needs
the workspace folder makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import builtins
import importlib
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel

# This file is tests/05_04_restart/test_restart.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs` in the empty folder, made from the sample code."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample code and the scripts the lesson runs."""
    root = tmp_path / "project"
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=root / "sample_code")
    (root / "demos").mkdir()
    for script in ("fresh_copy.py", "chat.py", "restart_session.py"):
        # Before the lesson there is no restart_session.py: the test that
        # runs it fails there, not every test that makes a copy.
        if (PROJECT_ROOT / "demos" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "demos" / script, dst=root / "demos")
    run_script(name="demos/fresh_copy.py", cwd=root)
    return root


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


def load_chat() -> ModuleType:
    """The chat loop, as a module, without running its main()."""
    return importlib.import_module("pydantic_tutor.chat.loop")


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


def messages_after(
    *, prompt: str, history: list[ModelMessage] | None = None
) -> list[ModelMessage]:
    """The messages one run on `echo` leaves, after `history`."""
    result = Agent(model=FunctionModel(function=echo)).run_sync(
        user_prompt=prompt, message_history=history
    )
    return result.all_messages()


def conversation_id(messages: list[ModelMessage]) -> str | None:
    return messages[-1].conversation_id


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
    fail_on: str | None = None,
) -> None:
    """Open the project's chat on a scripted model, saving to tmp_path/c.db.

    The model answers `I can see: [every prompt so far]`, and raises on the
    prompt `fail_on`. It streams, as the chat needs.
    """
    # The store's module is imported as a name of the package: before the
    # lesson it does not exist, and Ruff would sort a `from` import of it
    # apart from the rest of this block (I001).
    from pydantic_tutor.chat.agent import build_chat_agent
    from pydantic_tutor.composition import App
    from pydantic_tutor.stores import sqlite_conversations

    def check(messages: list[ModelMessage]) -> None:
        if fail_on is not None and said(messages)[-1] == fail_on:
            raise UnexpectedModelBehavior("the model sent nonsense")

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        check(messages)
        return echo(messages, info)

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        check(messages)
        yield f"I can see: {said(messages)}"

    workspace_root = tmp_path / "work" / "tinyjobs"
    if not workspace_root.exists():
        shutil.copytree(
            src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=workspace_root
        )
    monkeypatch.setattr("builtins.input", Keyboard(lines=lines))
    load_chat().main(
        app=App(
            agent=build_chat_agent(web_search=False, workspace_root=workspace_root),
            model=FunctionModel(function=reply, stream_function=stream),
            store=sqlite_conversations.SyncSqliteConversationStore(
                path=tmp_path / "c.db"
            ),
        )
    )


def test_a_new_process_carries_on_the_saved_conversation(project: Path) -> None:
    # The lesson's point: the chat is closed, a new process opens it, and
    # the model is sent the first process's turn.
    first = run_script(
        name="demos/restart_session.py", cwd=project, stdin="My name is Ada.\n\n"
    )
    assert "Turn 1. You've said: My name is Ada." in first
    assert "Carrying on" not in first
    second = run_script(
        name="demos/restart_session.py", cwd=project, stdin="What's my name?\n\n"
    )
    assert "Carrying on conversation " in second
    assert ": 2 messages. Type /new to start another.\n" in second
    assert "Turn 2. You've said: My name is Ada. | What's my name?" in second
    assert (project / "work" / "conversations.db").is_file()


def test_the_store_saves_the_messages_and_a_new_store_reads_them(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    messages = messages_after(prompt="hello")
    revision = SyncSqliteConversationStore(path=tmp_path / "c.db").save(
        messages=messages, revision=0
    )
    assert revision == 1
    saved = SyncSqliteConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    assert saved.revision == 1
    assert saved.messages == messages
    # The id is the one the messages carry: the run after a restart
    # continues the same conversation.
    assert conversation_id(saved.messages) == conversation_id(messages)
    after = messages_after(prompt="again", history=saved.messages)
    assert conversation_id(after) == conversation_id(messages)


def test_nothing_is_made_until_first_use_and_an_empty_store_has_none(
    tmp_path: Path,
) -> None:
    # Building the store opens nothing: the app builds one at startup,
    # and earlier lessons' tests build the app.
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    store = SyncSqliteConversationStore(path=tmp_path / "c.db")
    assert not (tmp_path / "c.db").exists()
    assert store.latest() is None


def test_there_is_nothing_to_save_in_no_messages(tmp_path: Path) -> None:
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    with pytest.raises(ValueError, match="no messages"):
        SyncSqliteConversationStore(path=tmp_path / "c.db").save(
            messages=[], revision=0
        )


def test_a_save_replaces_the_earlier_one_and_latest_is_the_last_saved(
    tmp_path: Path,
) -> None:
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    store = SyncSqliteConversationStore(path=tmp_path / "c.db")
    a = messages_after(prompt="a")
    b = messages_after(prompt="b")
    store.save(messages=a, revision=0)
    store.save(messages=b, revision=0)
    latest = store.latest()
    assert latest is not None
    assert latest.messages == b
    longer = messages_after(prompt="a2", history=a)
    assert store.save(messages=longer, revision=1) == 2
    latest = store.latest()
    assert latest is not None
    assert latest.messages == longer
    assert latest.revision == 2
    assert rows(path=tmp_path / "c.db") == 2


def test_a_stale_save_is_refused_and_the_file_keeps_the_first(tmp_path: Path) -> None:
    # Two readers of one saved conversation: the second to save would
    # overwrite the first's turn, so it is refused instead.
    from pydantic_tutor.stores.conversations import ConversationConflictError
    from pydantic_tutor.stores.sqlite_conversations import (
        SyncSqliteConversationStore,
    )

    store = SyncSqliteConversationStore(path=tmp_path / "c.db")
    store.save(messages=messages_after(prompt="one"), revision=0)
    first = store.latest()
    second = store.latest()
    assert first is not None
    assert second is not None
    one = messages_after(prompt="from first", history=first.messages)
    two = messages_after(prompt="from second", history=second.messages)
    store.save(messages=one, revision=first.revision)
    with pytest.raises(ConversationConflictError):
        store.save(messages=two, revision=second.revision)
    with pytest.raises(ConversationConflictError):
        # A conversation never saved cannot be saved as new twice either.
        store.save(messages=one, revision=0)
    latest = store.latest()
    assert latest is not None
    assert said(latest.messages) == ["one", "from first"]
    assert latest.revision == 2


def test_the_chat_saves_each_turn_and_new_starts_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    open_chat(monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["one", "two"])
    saved = SyncSqliteConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    assert said(saved.messages) == ["one", "two"]
    open_chat(
        monkeypatch=monkeypatch, tmp_path=tmp_path, lines=["three", "/new", "four"]
    )
    out = capsys.readouterr().out
    assert "A new conversation starts with your next line.\n" in out
    latest = SyncSqliteConversationStore(path=tmp_path / "c.db").latest()
    assert latest is not None
    assert said(latest.messages) == ["four"]
    assert conversation_id(latest.messages) != conversation_id(saved.messages)
    assert rows(path=tmp_path / "c.db") == 2


def test_two_chats_on_one_conversation_lose_no_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two chats open on the same saved conversation. The second saves a
    # turn while the first is waiting for a line; the first's next turn
    # is refused, and says so, instead of overwriting the second's.
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

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
    latest = SyncSqliteConversationStore(path=tmp_path / "c.db").latest()
    assert latest is not None
    assert said(latest.messages) == ["one", "from second"]


def test_a_turn_that_fails_is_not_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The turn raises out of the chat before the save: what is saved is
    # the conversation as it was.
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    with pytest.raises(UnexpectedModelBehavior):
        open_chat(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            lines=["one", "two"],
            fail_on="two",
        )
    saved = SyncSqliteConversationStore(path=tmp_path / "c.db").latest()
    assert saved is not None
    assert said(saved.messages) == ["one"]


def test_the_app_saves_to_the_file_the_settings_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Path
) -> None:
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    assert Settings().database == Path("pydantic-tutor.db")
    monkeypatch.setenv("PYDANTIC_TUTOR_DATABASE", str(tmp_path / "x.db"))
    app = build_app(settings=Settings(chat_model="test", web_search=False))
    from pydantic_tutor.stores.sqlite_conversations import (
        SyncSqliteConversationStore,
    )

    assert isinstance(app.store, SyncSqliteConversationStore)
    assert app.store.path == tmp_path / "x.db"
    assert not (tmp_path / "x.db").exists()  # made on first use, by the chat


def test_git_ignores_the_saved_conversations() -> None:
    lines = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert "/pydantic-tutor.db" in lines


def test_the_chat_keeps_its_conversations_in_any_store() -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from pydantic_tutor.stores.conversations import ConversationStore, Saved
    from pydantic_tutor.stores.sqlite_conversations import SyncSqliteConversationStore

    class InMemoryStore(ConversationStore):
        def __init__(self) -> None:
            self.kept: Saved | None = None

        def save(self, *, messages: list[ModelMessage], revision: int) -> int:
            self.kept = Saved(messages=messages, revision=revision + 1)
            return revision + 1

        def latest(self) -> Saved | None:
            return self.kept

    store = InMemoryStore()
    assert store.latest() is None
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hi")], conversation_id="c1")
    ]
    assert store.save(messages=messages, revision=0) == 1
    latest = store.latest()
    assert latest is not None and latest.messages == messages
    assert issubclass(SyncSqliteConversationStore, ConversationStore)
