"""Checks that the project from the seam lesson is in place.

Our checks for the seam lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done. None of them needs an API key or makes a request:
every run is on a scripted model.

Two tests were moved here: one pinned the codebase toolset on the
constructor's `toolsets=` (04-05), one pinned the prompt behind the two
capabilities the library adds of its own (04-04). Each says where it came
from, and proves its point as the agent is now built.

Two tests were moved on by the approval lesson (05-01), which adds two
capabilities to the stack, one of them declared innermost: that
`codebase_capabilities()` is new on each call and holds the whole stack, and
that the prompt comes first and the log last, are now proved in
`tests/05_01_approval/test_approval.py`, by
`test_codebase_capabilities_are_new_on_each_call_and_hold_the_whole_stack`
and `test_the_agent_takes_its_prompt_first_and_logs_after_the_cap`.

Three were moved on by the lesson on loading tools on demand (06-02),
which puts a catalog line in the instructions and `load_capability` among
the tools: that any order of the stack sends the same, what
`stack_order.py` prints, and that the agent takes the codebase toolset
as a capability are proved in `tests/06_02_deferred/test_deferred.py`, by
the tests of the same names.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/04_06_seam/test_seam.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def test_build_agent_builds_a_new_agent_on_what_only_the_constructor_sets() -> None:
    from pydantic_tutor.agent import agent, build_agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    first, second = build_agent(), build_agent()
    assert first is not second
    for built in (first, second, agent):
        assert built.name == "assistant"
        assert built.deps_type is CodebaseDeps
        assert built.output_type is str
        assert built.model_settings == {"timeout": 60}


def test_with_no_capabilities_the_agent_gives_the_model_nothing() -> None:
    # The lesson's first point: everything the model is given comes from
    # the stack. The constructor alone sends no instructions and no tools,
    # but its request settings still apply.
    from pydantic_tutor.agent import build_agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart(content="ok")])

    build_agent(capabilities=[]).run_sync(
        user_prompt="hello",
        model=FunctionModel(function=reply),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
    )
    assert not seen[0].model_request_parameters.instruction_parts
    assert seen[0].function_tools == []
    assert (seen[0].model_settings or {}).get("timeout") == 60


def test_the_prompt_and_the_log_declare_their_places() -> None:
    from pydantic_tutor.agent import PROMPT_PATH
    from pydantic_tutor.prompt_file import PromptFile
    from pydantic_tutor.request_log import request_log

    prompt_order = PromptFile(path=PROMPT_PATH).get_ordering()
    assert prompt_order is not None
    assert prompt_order.position == "outermost"
    log_order = request_log.get_ordering()
    assert log_order is not None
    assert log_order.position == "innermost"
