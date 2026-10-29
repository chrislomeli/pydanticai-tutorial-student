"""Checks that the project from the lesson on loading tools on demand is in place.

Our checks for the deferred lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

Seventeen checks were moved here from earlier lessons, because this lesson
defers the notebook: a request no longer carries its five tools or its
guidance, but a catalog line and `load_capability`. They pinned the number of
tools, the instructions' text, or a call to a notebook tool made before
any load: from the hooks lesson (04-02) one, from the memory lesson
(04-05) twelve, from the seam lesson (04-06) three, and from the approval
lesson (05-01) one case. Each says where it came from.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import logging
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import Memory

# This file is tests/06_02_deferred/test_deferred.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
LOGGER = "pydantic_tutor.capabilities.request_log"
# The chat's prompt since the console-chat lesson, word for word.
PROMPT = "You are a helpful assistant. Answer briefly."
WORKSPACE_NAME = "Your workspace holds the Python project tinyjobs."
FILE_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "find_files",
    "create_directory",
    "file_info",
]
SHELL_TOOLS = ["run_command", "start_command", "check_command", "stop_command"]
OUR_TOOLS = ["list_modules", "read_docstring"]
MEMORY_TOOLS = [
    "write_memory",
    "read_memory",
    "delete_memory",
    "search_memory",
    "list_memory",
]
# What the model may call before the load: all but the notebook's, and the
# tool that loads it.
SENT_TOOLS = len(FILE_TOOLS) + len(SHELL_TOOLS) + len(OUR_TOOLS) + 1
# What the request's definitions hold: the notebook's tools too.
DEFINED_TOOLS = SENT_TOOLS + len(MEMORY_TOOLS)


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


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs`, with a notebook that already holds two files."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    notes = root / ".agent-memory" / "main"
    notes.mkdir(parents=True)
    (notes / "MEMORY.md").write_text("- Prefers terse answers.\n")
    (notes / "logging.md").write_text("- Logging goes to stderr.\n")
    return root


def run_script(
    *, name: str, folder: Path, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / name)],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
        input=stdin,
        env={
            **env,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
        },
    )


class Recorder:
    """A model that records each request, making the calls it is given first."""

    def __init__(self, *, calls: list[ToolCallPart] | None = None) -> None:
        self.calls = list(calls or [])
        self.infos: list[AgentInfo] = []
        self.messages: list[list[ModelMessage]] = []

    def model(self) -> FunctionModel:
        return FunctionModel(function=self.reply)

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.messages.append(list(messages))
        if self.calls:
            return ModelResponse(parts=[self.calls.pop(0)])
        return ModelResponse(parts=[TextPart(content="ok")])

    def tools(self, *, request: int = 0) -> list[str]:
        """The names of the tools the model was sent with one request."""
        return [tool.name for tool in self.infos[request].function_tools]

    @property
    def instructions(self) -> str | None:
        return self.infos[0].instructions

    def pieces(self) -> list[str]:
        """The pieces of the first request's instructions, in order."""
        parts = self.infos[0].model_request_parameters.instruction_parts or []
        return [part.content for part in parts]

    @property
    def returned(self) -> str:
        """The last tool result, or refusal, the model was sent."""
        parts = [
            part
            for messages in self.messages
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart | RetryPromptPart)
        ]
        last = parts[-1]
        if isinstance(last, RetryPromptPart):
            return last.model_response()
        return last.model_response_str()


def call(*, tool_name: str, **args: str) -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args=args)


def load() -> ToolCallPart:
    """The call that loads the notebook."""
    return call(tool_name="load_capability", id="memory")


def catalog_line() -> str:
    from pydantic_tutor.chat.agent import MEMORY_DESCRIPTION

    return f"- memory: {MEMORY_DESCRIPTION}"


def run_chat(
    *,
    root: Path,
    recorder: Recorder,
    prompt: str = "hi",
    history: list[ModelMessage] | None = None,
) -> list[ModelMessage]:
    """Run the chat's agent once on `root`, on the recorder's model."""
    from pydantic_tutor.chat.agent import build_chat_agent

    chat = build_chat_agent(web_search=False, workspace_root=root)
    result = chat.run_sync(
        user_prompt=prompt, model=recorder.model(), message_history=history
    )
    return result.all_messages()


# Point 1: a deferred capability is a catalog line and one tool, until loaded.


def test_the_notebook_is_deferred_and_described(workspace: Path) -> None:
    from pydantic_tutor.chat.agent import MEMORY_DESCRIPTION, build_chat_agent

    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    [memory] = [c for c in chat.root_capability.capabilities if isinstance(c, Memory)]
    assert memory.defer_loading is True
    assert memory.id == "memory"
    assert memory.description == MEMORY_DESCRIPTION
    # The catalog is all the model reads about it until it loads it.
    assert "load" in MEMORY_DESCRIPTION


def test_before_the_load_a_request_carries_the_catalog_not_the_notebook(
    workspace: Path,
) -> None:
    # The lesson's point: the notebook's five tools and its guidance are not
    # sent. One line names it, and one tool loads it.
    from tutor_given.memory_text import LISTING_GUIDANCE

    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    sent = recorder.tools()
    assert "load_capability" in sent
    assert not set(MEMORY_TOOLS) & set(sent)
    assert len(sent) == SENT_TOOLS == 15
    assert recorder.instructions is not None
    assert recorder.instructions.endswith(catalog_line())
    assert LISTING_GUIDANCE not in recorder.instructions
    # The definitions the request holds still list the notebook's tools; the
    # model is not sent them.
    held = recorder.infos[0].model_request_parameters.function_tools
    assert len(held) == DEFINED_TOOLS == 20


def test_the_load_returns_the_guidance_and_the_tools_follow_on_the_next_request(
    workspace: Path,
) -> None:
    from tutor_given.memory_text import LISTING_GUIDANCE

    recorder = Recorder(calls=[load()])
    run_chat(root=workspace, recorder=recorder)
    assert len(recorder.infos) == 2  # the load costs a request
    # Not in the load's own request; in the next.
    assert not set(MEMORY_TOOLS) & set(recorder.tools(request=0))
    assert set(MEMORY_TOOLS) <= set(recorder.tools(request=1))
    assert len(recorder.tools(request=1)) == DEFINED_TOOLS
    # The guidance is the load's return, in the history; the instructions
    # never change.
    load_return = next(
        part
        for message in recorder.messages[1]
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "load_capability"
    )
    assert LISTING_GUIDANCE in load_return.model_response_str()
    assert recorder.infos[0].instructions == recorder.infos[1].instructions


def test_a_conversation_keeps_it_loaded_and_a_new_one_does_not(
    workspace: Path,
) -> None:
    first = Recorder(calls=[load()])
    history = run_chat(root=workspace, recorder=first)
    # The same conversation: the history holds the load, so the next turn
    # starts with the notebook's tools, and no second load.
    same = Recorder()
    run_chat(root=workspace, recorder=same, prompt="again", history=history)
    assert set(MEMORY_TOOLS) <= set(same.tools(request=0))
    # A new conversation, no history: hidden again.
    new = Recorder()
    run_chat(root=workspace, recorder=new, prompt="again")
    assert not set(MEMORY_TOOLS) & set(new.tools(request=0))


def test_a_hidden_tool_called_before_the_load_is_refused(workspace: Path) -> None:
    recorder = Recorder(
        calls=[call(tool_name="write_memory", content="- x\n", file="x.md")]
    )
    run_chat(root=workspace, recorder=recorder)
    assert "not available yet" in recorder.returned
    assert "load_capability" in recorder.returned
    assert not (workspace / ".agent-memory" / "main" / "x.md").exists()


def test_once_loaded_the_notebook_writes_without_asking(workspace: Path) -> None:
    # Moved from 05-01 (reads_and_the_notebook_do_not_ask): the notebook's
    # write needs a load first, and then is run, not held.
    recorder = Recorder(
        calls=[
            load(),
            call(tool_name="write_memory", content="- a note\n", file="a.md"),
        ]
    )
    run_chat(root=workspace, recorder=recorder)
    assert '"status":"created"' in recorder.returned
    assert (workspace / ".agent-memory" / "main" / "a.md").exists()


# Point 2: what the request costs, before and after.


def turn_lines(*, out: str) -> list[tuple[int, int, int]]:
    """Per request of the session: (instructions, tool definitions, tools) in tokens and count."""
    found = re.findall(
        r"instructions ([\d,]+) · tool definitions ([\d,]+) \((\d+)\)", out
    )
    return [
        (int(a.replace(",", "")), int(b.replace(",", "")), int(c)) for a, b, c in found
    ]


def session(*, project: Path) -> str:
    lines = "What is in tinyjobs?\nRemember that we log to stderr.\nAnd again?\n"
    result = run_script(name="cost_session.py", folder=project, stdin=lines)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_cost_session_sends_fewer_tools_until_the_notebook_is_loaded(
    project: Path,
) -> None:
    out = session(project=project)
    requests = turn_lines(out=out)
    assert len(requests) == 9  # three requests a turn, three turns
    before, after = requests[:4], requests[4:]
    assert {count for _, _, count in before} == {16}
    assert {count for _, _, count in after} == {21}
    assert len({instructions for instructions, _, _ in requests}) == 1
    assert before[0][1] < after[0][1]  # more tool definitions once loaded


def test_cost_session_loads_in_the_second_turn_and_keeps_it(project: Path) -> None:
    out = session(project=project)
    assert "agent> Noted." in out
    sent = [
        int(n.replace(",", ""))
        for n in re.findall(r"this turn sent ([\d,]+) tokens", out)
    ]
    assert len(sent) == 3
    # The turn that loads sends more than the one before; the third, with
    # the notebook's tools in every request, more than the first.
    assert sent[1] > sent[0]
    assert sent[2] > sent[0]
    assert (
        re.findall(r"this turn sent [\d,]+ tokens in (\d+) requests", out) == ["3"] * 3
    )
    notebook = project / "work" / "tinyjobs" / ".agent-memory" / "main"
    assert any(
        path.read_text().strip() == "- we log to stderr."
        for path in notebook.glob("*.md")
    )


def test_the_log_counts_only_what_the_model_may_call(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The hook's own view holds the hidden tools; the log counts the
    # available ones, which is what the model is sent.
    from pydantic_tutor.chat.agent import build_chat_agent

    seen: list[int] = []

    async def before(
        ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        seen.append(len(request_context.model_request_parameters.function_tools))
        return request_context

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder()
    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    chat.run_sync(
        user_prompt="hi",
        model=recorder.model(),
        capabilities=[Hooks(before_model_request=before)],
    )
    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert seen == [DEFINED_TOOLS]
    assert logged == [f"request step=1 messages=1 tools={SENT_TOOLS}"]
    assert len(recorder.infos[0].function_tools) == SENT_TOOLS


# Moved: they pinned the chat's tools, instructions or log before the deferral.


def test_the_log_counts_the_tools_actually_sent(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Moved from 04-02: with the tools hidden but ours, the model is sent
    # those and `load_capability`, and the log says so.
    from pydantic_tutor.chat.agent import build_chat_agent

    def only_ours(
        ctx: RunContext[None], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            tool
            for tool in tool_defs
            if tool.name in {"list_modules", "read_docstring", "load_capability"}
        ]

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder(calls=[call(tool_name="list_modules")])
    chat = build_chat_agent(web_search=False, workspace_root=workspace)
    chat.run_sync(
        user_prompt="x",
        model=recorder.model(),
        capabilities=[Hooks(prepare_tools=only_ours)],
    )
    assert [len(info.function_tools) for info in recorder.infos] == [3, 3]
    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert logged == [
        "request step=1 messages=1 tools=3",
        "request step=2 messages=3 tools=3",
    ]


def test_no_request_carries_the_notebook_and_the_model_lists_it(
    workspace: Path,
) -> None:
    # Moved from 04-05: the model loads the notebook first.
    recorder = Recorder(calls=[load(), call(tool_name="list_memory")])
    run_chat(root=workspace, recorder=recorder)
    for messages in recorder.messages:
        for message in messages:
            for part in message.parts:
                text = str(getattr(part, "content", ""))
                assert "<memory>" not in text
                assert "Prefers terse answers" not in text
    [first, *_] = recorder.messages
    assert [part.part_kind for part in first[0].parts] == ["user-prompt"]
    assert recorder.returned == '{"files":["MEMORY.md","logging.md"],"truncated":false}'


def test_a_note_written_in_one_run_is_found_in_the_next(workspace: Path) -> None:
    # Moved from 04-05: each run loads the notebook, having no history.
    shutil.rmtree(workspace / ".agent-memory")
    note = "- Logging goes to stderr.\n"
    run_chat(
        root=workspace,
        recorder=Recorder(
            calls=[
                load(),
                call(tool_name="write_memory", content=note, file="logging.md"),
            ]
        ),
    )
    assert (workspace / ".agent-memory" / "main" / "logging.md").read_text() == note
    later = Recorder(
        calls=[
            load(),
            call(tool_name="list_memory"),
            call(tool_name="read_memory", file="logging.md"),
        ]
    )
    run_chat(root=workspace, recorder=later)
    assert len(later.messages[0]) == 1
    assert later.returned == note


def test_list_memory_names_only_files_read_memory_takes(workspace: Path) -> None:
    # Moved from 04-05.
    notes = workspace / ".agent-memory" / "main"
    (notes / "team").mkdir()
    (notes / "team" / "deep.md").write_text("x\n")
    (notes / "draft.txt").write_text("x\n")
    recorder = Recorder(calls=[load(), call(tool_name="list_memory")])
    run_chat(root=workspace, recorder=recorder)
    assert recorder.returned == '{"files":["MEMORY.md","logging.md"],"truncated":false}'


def test_notebook_prints_two_sessions(project: Path) -> None:
    # Moved from 04-05: each session loads the notebook first.
    result = run_script(name="notebook.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "session: Remember that we log to stderr."
    assert lines[1:3] == [
        "  request ['user-prompt']",
        '  call   load_capability {"id": "memory"}',
    ]
    assert lines[3] == "  request ['tool-return', 'user-prompt']"
    assert lines[4].startswith('  result {"instructions":"You have a persistent memory')
    assert lines[5] == (
        '  call   write_memory {"content": "- Logging goes to stderr, so stdout '
        'stays pipeable.\\n", "file": "logging.md"}'
    )
    assert '"status":"created"' in lines[7]
    assert (
        lines[8:11]
        == [
            "  output Noted.",
            "session: What did we decide about logging?",
            "  request ['user-prompt']",
        ][:3]
    )
    assert lines[11] == '  call   load_capability {"id": "memory"}'
    assert "  call   list_memory {}" in lines
    assert lines[-1] == (
        "  output We decided: - Logging goes to stderr, so stdout stays pipeable."
    )
    notebook = project / "work" / "tinyjobs" / ".agent-memory" / "main" / "logging.md"
    assert (
        notebook.read_text() == "- Logging goes to stderr, so stdout stays pipeable.\n"
    )


def test_the_chat_sends_the_prompt_the_workspace_then_the_catalog(
    workspace: Path,
) -> None:
    # Moved from 04-05 (guidance_is_sent_after_the_prompt...): the notebook's
    # guidance is no longer instructions; the catalog line, computed for each
    # request, comes after the static text.
    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    parts = recorder.infos[0].model_request_parameters.instruction_parts or []
    assert [part.content for part in parts[:2]] == [PROMPT, WORKSPACE_NAME]
    assert parts[2].content.endswith(catalog_line())
    assert [part.dynamic for part in parts] == [False, False, True]


def test_the_chat_is_offered_the_toolset_and_the_tools_it_had(
    workspace: Path,
) -> None:
    # Moved from 04-05: the chat's tools, the notebook's five held back, and
    # `load_capability` added.
    from pydantic_tutor.composition import build_app
    from pydantic_tutor.settings import Settings

    recorder = Recorder()
    app = build_app(
        settings=Settings(chat_model="test", web_search=False, workspace_root=workspace)
    )
    app.agent.run_sync(user_prompt="hi", model=recorder.model())
    assert sorted(recorder.tools()) == sorted(
        [*FILE_TOOLS, *SHELL_TOOLS, *OUR_TOOLS, "load_capability"]
    )
    assert len(recorder.tools()) == SENT_TOOLS


def test_the_chat_sends_the_prompt_then_the_toolsets_then_the_catalog(
    workspace: Path,
) -> None:
    # Moved from 04-05.
    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    assert recorder.pieces() == [PROMPT, WORKSPACE_NAME, recorder.pieces()[2]]
    assert (
        recorder.instructions
        == f"{PROMPT}\n\n{WORKSPACE_NAME}\n\n" + (recorder.pieces()[2])
    )


def test_the_prompt_file_is_still_what_the_chat_is_sent_first(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Moved from 04-05: change the file, and the chat built next is sent the
    # change, ahead of the rest.
    from pydantic_tutor.chat import agent as chat_agent

    edited = tmp_path / "chat.md"
    edited.write_text("You are a pirate.\n", encoding="utf-8")
    monkeypatch.setattr(target=chat_agent, name="PROMPT_PATH", value=edited)
    recorder = Recorder()
    run_chat(root=workspace, recorder=recorder)
    assert recorder.pieces()[:2] == ["You are a pirate.", WORKSPACE_NAME]


def test_workspace_reach_counts_the_tools_the_model_is_offered(project: Path) -> None:
    # Moved from 04-05, where it printed 19: the notebook's five are held.
    result = run_script(name="workspace_reach.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert f"tools the model is offered: {SENT_TOOLS}" in lines
    assert "README.md: # tinyjobs" in lines


def test_toolset_views_prints_the_chat_and_what_each_wrapper_sends(
    project: Path,
) -> None:
    # Moved from 04-05: the chat's line ends with ours, not the notebook's.
    result = run_script(name="toolset_views.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    [chat_tools] = [
        line for line in lines if line.startswith("  tools: load_capability, ")
    ]
    assert ", list_modules, read_docstring" in chat_tools
    assert not any(name in chat_tools for name in MEMORY_TOOLS)
    assert (
        "  read_docstring('tinyjobs.jobs') returned: "
        "The job record and the queue that holds jobs." in lines
    )
    assert "  tools: list_modules, read_docstring" in lines
    assert "  tools: workspace_list_modules, workspace_read_docstring" in lines
    assert "  tools: list_modules" in lines
    assert (
        "  the model got back: There is no module 'jobs'. "
        "Call list_modules to see them." in lines
    )


def test_each_request_is_logged_with_its_step_messages_and_tools(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Moved from 04-05, where the log said tools=19.
    caplog.set_level(level=logging.INFO, logger=LOGGER)
    recorder = Recorder(calls=[call(tool_name="list_modules")])
    run_chat(root=workspace, recorder=recorder)
    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert logged == [
        f"request step=1 messages=1 tools={SENT_TOOLS}",
        f"request step=2 messages=3 tools={SENT_TOOLS}",
    ]
    assert [len(info.function_tools) for info in recorder.infos] == [SENT_TOOLS] * 2


def test_watch_requests_prints_the_log(project: Path) -> None:
    # Moved from 04-05.
    result = run_script(name="watch_requests.py", folder=project)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == [
        f"{LOGGER}: request step=1 messages=1 tools={SENT_TOOLS}",
        f"{LOGGER}: request step=2 messages=3 tools={SENT_TOOLS}",
    ]
    assert lines[2].startswith("output: ")
    assert lines[3].startswith("usage: ")


class StyleGuide(AbstractCapability[None]):
    """A capability that adds text and declares no place."""

    def get_instructions(self) -> str:
        return "Use plain words."


def run_built(
    *,
    root: Path,
    capabilities: list[AgentCapability[None]],
    caplog: pytest.LogCaptureFixture,
) -> tuple[Recorder, list[int]]:
    """Build the chat's agent with `capabilities`, run it once; return the
    model's view and the message counts the request log logged."""
    from pydantic_tutor.chat.agent import build_chat_agent

    caplog.clear()
    recorder = Recorder(calls=[call(tool_name="list_modules")])
    build_chat_agent(
        web_search=False, workspace_root=root, capabilities=capabilities
    ).run_sync(user_prompt="What does this project do?", model=recorder.model())
    logged = [
        int(record.getMessage().split("messages=")[1].split()[0])
        for record in caplog.records
        if record.name == LOGGER
    ]
    return recorder, logged


def test_any_order_of_the_stack_sends_the_prompt_first_and_logs_what_is_sent(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Moved from 04-06: with a capability that changes the request added
    # last, in every order tried, the prompt comes first, the same three
    # pieces and the same 15 tools are sent, and the log counts the messages
    # the model is sent.
    from pydantic_tutor.chat.agent import chat_capabilities
    from tutor_given.stand_ins import KeepLastTwo as KeepTwo

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    shuffle = random.Random(4).shuffle
    orders: list[list[AgentCapability[None]]] = []
    for index in range(2 + 12):
        stack: list[AgentCapability[None]] = [
            *chat_capabilities(web_search=False, workspace_root=workspace),
            KeepTwo(),
        ]
        if index == 1:
            stack.reverse()
        elif index > 1:
            shuffle(stack)
        orders.append(stack)
    for stack in orders:
        recorder, logged = run_built(root=workspace, capabilities=stack, caplog=caplog)
        pieces = recorder.pieces()
        assert pieces[0] == PROMPT
        assert len(pieces) == 3
        assert {PROMPT, WORKSPACE_NAME} <= set(pieces)
        assert pieces[2].endswith(catalog_line())
        names = [
            sorted(tool.name for tool in info.function_tools) for info in recorder.infos
        ]
        assert all(len(n) == SENT_TOOLS for n in names)
        assert names[0] == names[1]
        assert [len(m) for m in recorder.messages] == [1, 2]
        assert logged == [1, 2]


def test_undeclared_text_and_tools_follow_the_list(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Moved from 04-06: what a declaration does not cover follows the list.
    # A text capability comes before the toolset's sentence wherever it is
    # listed (a toolset's text comes last), and the catalog, computed for
    # each request, after all of them. The prompt is first either way.
    from pydantic_tutor.chat.agent import chat_capabilities

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    style = "Use plain words."
    for listed in ("before", "after"):
        stack = chat_capabilities(web_search=False, workspace_root=workspace)
        listing = (
            [StyleGuide(), *stack] if listed == "before" else [*stack, StyleGuide()]
        )
        pieces = run_built(root=workspace, capabilities=listing, caplog=caplog)[
            0
        ].pieces()
        assert pieces[:3] == [PROMPT, style, WORKSPACE_NAME]
        assert pieces[3].endswith(catalog_line())
    # Tools are offered in list order too: reversed, ours come first, and
    # `load_capability` is still first of all.
    backwards = chat_capabilities(web_search=False, workspace_root=workspace)[::-1]
    reversed_run = run_built(root=workspace, capabilities=backwards, caplog=caplog)[0]
    assert reversed_run.tools()[:3] == [
        "load_capability",
        "list_modules",
        "read_docstring",
    ]


def test_stack_order_prints_both_orders_logged_right(project: Path) -> None:
    # Moved from 04-06: the log's count is the 15 the model may call.
    made = run_script(name="stack_order.py", folder=project)
    assert made.returncode == 0, made.stderr
    out = made.stdout.splitlines()
    for label in ("forwards", "reversed"):
        at = out.index(label)
        block = out[at + 1 : at + 8]
        assert block[0] == f"{LOGGER}: request step=1 messages=1 tools={SENT_TOOLS}"
        assert block[1] == "  model: sent 1 message(s)"
        assert block[2].startswith("    instructions: You are a helpful")
        assert block[5] == f"{LOGGER}: request step=2 messages=2 tools={SENT_TOOLS}"
        assert block[6] == "  model: sent 2 message(s)"
