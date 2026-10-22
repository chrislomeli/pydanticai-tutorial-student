"""Which of the agent's tool calls wait for a person, and how they are answered.

The mechanism is the library's: the harness's `ToolGuardrail` holds a call
for approval before it runs, and a `HandleDeferredToolCalls` capability
answers it inside the run. This module holds the project's policy: which
tools ask (`ask_first`), and the answers (`refuse_unattended`,
`approve_all`), given to a run by `approver(...)`.
"""

from collections.abc import Callable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness import GuardrailResult
from pydantic_ai_harness.filesystem import FILE_SYSTEM_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from pydantic_ai_harness.guardrails import ToolCallInfo
from pydantic_ai_harness.shell import SHELL_TOOL_NAMES

from pydantic_tutor.repo_tools import RepoDeps

# The harness's tools that change the repository or start a process, by
# its own lists, so a tool it can add (`Shell`'s opt-in `shell`) asks too:
# every file tool it does not call read-only, and every shell tool but the
# two that check on or stop a command already started. Our own tools, the
# notebook and the reads run without asking.
ASK_FIRST = (frozenset(FILE_SYSTEM_TOOL_NAMES) - READ_ONLY_TOOL_NAMES) | (
    frozenset(SHELL_TOOL_NAMES) - {"check_command", "stop_command"}
)

# What the model is told about a call no one was there to approve.
UNATTENDED = (
    "Not run: no one was there to approve it. Tell the user what it would have done."
)

# The `id` every answerer has. A run given an answerer replaces the agent's
# for that run, because they share it.
APPROVER_ID = "approver"

# A function that answers the calls held for approval: for each one,
# `ToolApproved()` (or `True`) to run it, or `ToolDenied(message=...)` to
# refuse it, the message being what the model is told.
Answer = Callable[[RunContext[RepoDeps], DeferredToolRequests], DeferredToolResults]


def ask_first(call: ToolCallInfo) -> GuardrailResult:
    """Hold a call to an ASK_FIRST tool for approval; let any other run."""
    if call.name in ASK_FIRST:
        return GuardrailResult.approve()
    return GuardrailResult.allow()


def refuse_unattended(
    ctx: RunContext[RepoDeps], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Refuse every call held for approval: no one is there to approve it."""
    return DeferredToolResults(
        approvals={
            call.tool_call_id: ToolDenied(message=UNATTENDED)
            for call in requests.approvals
        }
    )


def approve_all(
    ctx: RunContext[RepoDeps], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Approve every call held for approval.

    Only for a run whose calls were chosen in advance, such as a script's
    stand-in model; never the agent's own answer.
    """
    return requests.build_results(approve_all=True)


def approver(*, answer: Answer) -> HandleDeferredToolCalls[RepoDeps]:
    """A capability that answers the calls held for approval with `answer`."""
    return HandleDeferredToolCalls(handler=answer, id=APPROVER_ID)
