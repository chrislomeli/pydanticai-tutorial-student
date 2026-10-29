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

Three checks moved to the deferred lesson (06-02): they pinned the notebook's
guidance among the instructions and its tools in every request.

Two checks moved to the approval lesson (05-01), which adds two capabilities
to the chat: that its capabilities are built anew and are eight, and that the
prompt is first and the log last.
"""

import os
import shutil
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_06_seam/test_seam.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PREFIX = "PYDANTIC_TUTOR_"
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
PROMPT_FILE: Path = PROJECT_ROOT / "src" / "pydantic_tutor" / "prompts" / "chat.md"
# The chat's prompt since the console-chat lesson, word for word.
PROMPT = "You are a helpful assistant. Answer briefly."


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


# Moved: it pinned the prompt behind the library's own capabilities.
