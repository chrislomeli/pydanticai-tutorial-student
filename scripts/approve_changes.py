"""A person approves one change and refuses another: what runs, and what the model is told.

From the approval lesson (05-01). Runs the agent once on `work/tinyjobs`,
asked for a `--verbose` flag and a test for it. A stand-in model asks, in
one response, to write a test file and to run the tests; a stand-in
person, `at_the_desk`, is shown each call held for approval, approves the
write and refuses the command. The model prints what it got back for each
call. Run `scripts/fresh_copy.py` first: the test file is written into the
copy. Kept working: a later lesson that changes how calls are approved
updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)

from pydantic_tutor.agent import agent
from pydantic_tutor.approval import approver
from pydantic_tutor.repo_tools import RepoDeps

REPO = Path("work/tinyjobs")
TEST_PATH = "tests/test_verbose.py"
# The test the stand-in model writes. It fails until the flag exists.
TEST = (
    "from tinyjobs.cli import build_parser\n\n\n"
    "def test_verbose() -> None:\n"
    '    assert build_parser().parse_args(["--verbose", "list"]).verbose\n'
)


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: ask to write a test and run it, then say what came back."""
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="write_file", args={"path": TEST_PATH, "content": TEST}
                ),
                ToolCallPart(
                    tool_name="run_command", args={"command": f"pytest -q {TEST_PATH}"}
                ),
            ]
        )
    for part in returns:
        print(f"  model got: {part.tool_name}, {part.outcome}: {part.content}")
    return ModelResponse(parts=[TextPart(content="Done.")])


def at_the_desk(
    ctx: RunContext[RepoDeps], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Stand in for a person: approve the write, refuse the command."""
    answers: dict[str, ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        print(f"  asked: {call.tool_name} {call.args_as_dict()}")
        if call.tool_name == "write_file":
            answers[call.tool_call_id] = ToolApproved()
        else:
            answers[call.tool_call_id] = ToolDenied(
                message="Not now: I'll read the test before anything runs."
            )
    return requests.build_results(approvals=answers)


agent.run_sync(
    user_prompt="Add a --verbose flag to the CLI, and a test for it.",
    model=FunctionModel(function=reply),
    deps=RepoDeps(root=REPO),
    # Replaces the agent's answerer, for this run only: they share an `id`.
    capabilities=[approver(answer=at_the_desk)],
)
print(f"{TEST_PATH} written: {(REPO / TEST_PATH).exists()}")
