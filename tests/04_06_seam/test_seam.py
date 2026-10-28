"""Checks that the project from the seam lesson is in place.

Our checks for the seam lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a stand-in model.

One check was moved here from the prompt lesson (04-04): it pinned the
prompt behind the two capabilities the library adds of its own, and the
prompt now declares its place, ahead of them.

Each test runs in an empty folder of its own, so the project's own `.env`
and `work/` folder are never read. A test that needs the workspace folder
makes `work/tinyjobs` there, from the sample code.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.

Two checks moved to the approval lesson (05-01), which adds two capabilities
to the chat: that its capabilities are built anew and are eight, and that the
prompt is first and the log last.
"""

import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_06_seam/test_seam.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
PROMPT_FILE: Path = PROJECT_ROOT / "src" / "pydantic_tutor" / "prompts" / "chat.md"
LOGGER = "pydantic_tutor.capabilities.request_log"
# The chat's prompt since the console-chat lesson, word for word.
PROMPT = "You are a helpful assistant. Answer briefly."
WORKSPACE_NAME = "Your workspace holds the Python project tinyjobs."
ALL_TOOLS = 19  # the file tools 8, the shell's 4, ours 2, the notebook's 5
# How many orders of the stack the order test tries, beyond the two the
# lesson shows: enough to move every capability, few enough to be quick.
SHUFFLES = 12


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() in KEY_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """`work/tinyjobs`, a copy of the sample code."""
    root = tmp_path / "work" / "tinyjobs"
    shutil.copytree(src=PROJECT_ROOT / "sample_code" / "tinyjobs", dst=root)
    return root


@pytest.fixture
def project(tmp_path: Path, workspace: Path) -> Path:
    """A stand-in project folder: the sample code and `work/tinyjobs`."""
    shutil.copytree(src=PROJECT_ROOT / "sample_code", dst=tmp_path / "sample_code")
    shutil.copy(src=PROJECT_ROOT / "pyproject.toml", dst=tmp_path / "pyproject.toml")
    return tmp_path


class Recorder:
    """A model that lists the modules once, then answers; records each request."""

    def __init__(self) -> None:
        self.infos: list[AgentInfo] = []
        self.sizes: list[int] = []

    def model(self) -> FunctionModel:
        return FunctionModel(function=self.reply)

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.sizes.append(len(messages))
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content="ok")])
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])

    def instructions(self) -> list[str]:
        """The pieces of the first request's instructions, in order."""
        parts = self.infos[0].model_request_parameters.instruction_parts or []
        return [part.content for part in parts]


def run_built(
    *,
    root: Path,
    capabilities: list[AgentCapability[None]] | None,
    caplog: pytest.LogCaptureFixture,
) -> tuple[Recorder, list[int]]:
    """Build the chat's agent with `capabilities`, run it once; return the
    model's view and the message counts the request log logged."""
    from pydantic_tutor.chat.agent import build_chat_agent

    caplog.clear()
    recorder = Recorder()
    build_chat_agent(
        web_search=False, workspace_root=root, capabilities=capabilities
    ).run_sync(user_prompt="What does this project do?", model=recorder.model())
    logged = [
        int(record.getMessage().split("messages=")[1].split()[0])
        for record in caplog.records
        if record.name == LOGGER
    ]
    return recorder, logged


# Point 1: the constructor sets what no capability can; the stack holds the rest.


def test_the_constructor_sets_the_name_the_types_and_the_request_defaults(
    workspace: Path,
) -> None:
    from pydantic_tutor.chat.agent import build_chat_agent, chat_agent

    first = build_chat_agent(web_search=False, workspace_root=workspace)
    second = build_chat_agent(web_search=False, workspace_root=workspace)
    assert first is not second
    for built in (first, second, chat_agent):
        assert built.name == "chat"
        assert built.output_type is str
        assert built.model_settings == {"timeout": 120}


def test_with_no_capabilities_the_agent_gives_the_model_nothing(
    workspace: Path,
) -> None:
    # The lesson's first point: everything the model is given comes from
    # the stack. The constructor alone sends no instructions and no tools,
    # but its request defaults still apply.
    from pydantic_tutor.chat.agent import build_chat_agent

    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    build_chat_agent(
        web_search=True, workspace_root=workspace, capabilities=[]
    ).run_sync(user_prompt="hello", model=FunctionModel(function=reply))
    assert not seen[0].model_request_parameters.instruction_parts
    assert seen[0].function_tools == []
    assert (seen[0].model_settings or {}).get("timeout") == 120


def test_an_alternative_list_replaces_the_stack(workspace: Path) -> None:
    from pydantic_tutor.capabilities.prompt_file import PromptFile
    from pydantic_tutor.chat.agent import (
        PROMPT_PATH,
        build_chat_agent,
        chat_capabilities,
    )

    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    # Only the prompt: one piece of instructions, no tool.
    only_prompt: list[AgentCapability[None]] = [PromptFile(path=PROMPT_PATH)]
    build_chat_agent(
        web_search=True, workspace_root=workspace, capabilities=only_prompt
    ).run_sync(user_prompt="hi", model=FunctionModel(function=reply))
    parts = seen[0].model_request_parameters.instruction_parts or []
    assert [part.content for part in parts] == [PROMPT]
    assert seen[0].function_tools == []
    # Given no list, the agent is built from chat_capabilities(...).
    full = build_chat_agent(web_search=False, workspace_root=workspace)
    assert len(full.root_capability.capabilities) >= len(
        chat_capabilities(web_search=False, workspace_root=workspace)
    )


# Point 2: a capability whose place matters declares it.


def test_the_prompt_and_the_log_declare_their_places() -> None:
    from pydantic_tutor.capabilities.prompt_file import PromptFile
    from pydantic_tutor.capabilities.request_log import request_log
    from pydantic_tutor.chat.agent import PROMPT_PATH

    prompt_order = PromptFile(path=PROMPT_PATH).get_ordering()
    assert prompt_order is not None
    assert prompt_order.position == "outermost"
    log_order = request_log.get_ordering()
    assert log_order is not None
    assert log_order.position == "innermost"


def test_any_order_of_the_stack_sends_the_prompt_first_and_logs_what_is_sent(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The lesson's second point. With a capability that changes the
    # request added last, in every order tried: the prompt comes first,
    # the same three pieces and the same 19 tools are sent, and the log
    # counts the messages the model is sent.
    from pydantic_tutor.chat.agent import chat_capabilities
    from tutor_given.memory_text import LISTING_GUIDANCE
    from tutor_given.stand_ins import KeepLastTwo

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    shuffle = random.Random(4).shuffle  # fixed: the same orders every run
    orders: list[list[AgentCapability[None]]] = []
    for index in range(2 + SHUFFLES):
        stack: list[AgentCapability[None]] = [
            *chat_capabilities(web_search=False, workspace_root=workspace),
            KeepLastTwo(),
        ]
        if index == 1:
            stack.reverse()
        elif index > 1:
            shuffle(stack)
        orders.append(stack)
    for stack in orders:
        recorder, logged = run_built(root=workspace, capabilities=stack, caplog=caplog)
        pieces = recorder.instructions()
        assert pieces[0] == PROMPT
        assert sorted(pieces) == sorted([PROMPT, LISTING_GUIDANCE, WORKSPACE_NAME])
        names = [
            sorted(tool.name for tool in info.function_tools) for info in recorder.infos
        ]
        assert all(len(n) == ALL_TOOLS for n in names)
        assert names[0] == names[1]
        assert recorder.sizes == [1, 2]
        assert logged == recorder.sizes


class StyleGuide(AbstractCapability[None]):
    """A capability that adds text and declares no place."""

    def get_instructions(self) -> str:
        return "Use plain words."


def test_undeclared_text_and_tools_follow_the_list(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # What a declaration does not cover still follows the list: a text
    # capability listed before the notebook comes before its guidance, and
    # after it when listed after. The prompt is first either way.
    from pydantic_tutor.chat.agent import chat_capabilities
    from tutor_given.memory_text import LISTING_GUIDANCE

    caplog.set_level(level=logging.INFO, logger=LOGGER)
    style = "Use plain words."
    stack = chat_capabilities(web_search=False, workspace_root=workspace)
    before = run_built(
        root=workspace, capabilities=[StyleGuide(), *stack], caplog=caplog
    )[0]
    assert before.instructions() == [PROMPT, style, LISTING_GUIDANCE, WORKSPACE_NAME]
    stack = chat_capabilities(web_search=False, workspace_root=workspace)
    after = run_built(
        root=workspace, capabilities=[*stack, StyleGuide()], caplog=caplog
    )[0]
    assert after.instructions() == [PROMPT, LISTING_GUIDANCE, style, WORKSPACE_NAME]
    # Tools are offered in list order too: reversed, the notebook's come first.
    backwards = chat_capabilities(web_search=False, workspace_root=workspace)[::-1]
    reversed_run = run_built(root=workspace, capabilities=backwards, caplog=caplog)[0]
    offered = [tool.name for tool in reversed_run.infos[0].function_tools]
    assert offered[0].endswith("_memory")


def test_stack_order_prints_both_orders_logged_right(project: Path) -> None:
    made = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "demos" / "stack_order.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=project,
        env={
            **{k: v for k, v in os.environ.items() if k not in KEY_VARIABLES},
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYDANTIC_AI_NO_BANNER": "1",
        },
    )
    assert made.returncode == 0, made.stderr
    out = made.stdout.splitlines()
    # Each order: the log's count matches the model's, on both requests.
    for label in ("forwards", "reversed"):
        at = out.index(label)
        block = out[at + 1 : at + 8]
        assert block[0] == f"{LOGGER}: request step=1 messages=1 tools=19"
        assert block[1] == "  model: sent 1 message(s)"
        assert block[2].startswith("    instructions: You are a helpful")
        assert block[5] == f"{LOGGER}: request step=2 messages=2 tools=19"
        assert block[6] == "  model: sent 2 message(s)"


# Moved: it pinned the prompt behind the library's own capabilities.
