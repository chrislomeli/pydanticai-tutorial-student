"""A person approves one change and refuses another.

From the approval lesson (05-01). A given stand-in model asks to write a test
and run it; `at_the_desk` approves the write and refuses the command. Needs no
API key. Run `demos/fresh_copy.py` first. Kept working: a later lesson that
changes how calls are approved updates this script, or removes it and says why.
"""

from pydantic_ai import RunContext
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)

from pydantic_tutor.capabilities.approval import approver
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import TEST_PATH, asks_for_a_test_and_its_run


def at_the_desk(
    ctx: RunContext[None], requests: DeferredToolRequests
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


root = Settings().workspace_root
chat = build_chat_agent(web_search=False, workspace_root=root)
chat.run_sync(
    user_prompt="Add a --verbose flag to the CLI, and a test for it.",
    model=asks_for_a_test_and_its_run(),
    # Replaces the chat's own answerer for this run only: they share an `id`.
    capabilities=[approver(answer=at_the_desk)],
)
print(f"{TEST_PATH} written: {(root / TEST_PATH).exists()}")
