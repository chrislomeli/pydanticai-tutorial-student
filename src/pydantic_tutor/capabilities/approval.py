"""Which of the chat's tool calls wait for a person, and how they are answered.

The mechanism is the library's: the harness's `ToolGuardrail` holds a call
for approval before it runs, and a `HandleDeferredToolCalls` capability
answers it inside the run. This module holds the project's policy: which
tools ask (`ask_first`), and the answers, given to a run by `approver(...)`:
`refuse_unattended` is the agent's own, `at_the_keyboard` is the chat's.
"""

from collections.abc import Callable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai_harness import GuardrailResult
from pydantic_ai_harness.filesystem import FILE_SYSTEM_TOOL_NAMES, READ_ONLY_TOOL_NAMES
from pydantic_ai_harness.guardrails import ToolCallInfo
from pydantic_ai_harness.shell import SHELL_TOOL_NAMES

# The harness's tools that change the workspace or start a process, by its
# own lists, so a tool it can add (`Shell`'s opt-in `shell`) asks too: every
# file tool it does not call read-only, and every shell tool but the two
# that check on or stop a command already started. Our own tools, the
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
# `ToolApproved()` to run it, or `ToolDenied(message=...)` to refuse it, the
# message being what the model is told.
Answer = Callable[[RunContext[None], DeferredToolRequests], DeferredToolResults]


def ask_first(call: ToolCallInfo) -> GuardrailResult:
    """Hold a call to an ASK_FIRST tool for approval; let any other run."""
    if call.name in ASK_FIRST:
        return GuardrailResult.approve()
    return GuardrailResult.allow()


def refuse_unattended(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Refuse every call held for approval: no one is there to approve it."""
    return DeferredToolResults(
        approvals={
            call.tool_call_id: ToolDenied(message=UNATTENDED)
            for call in requests.approvals
        }
    )


def at_the_keyboard(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Show each held call and ask: it runs only if the person types `y`.

    Anything else refuses it, nothing typed and the end of input included,
    and what was typed is what the model is told.
    """
    answers: dict[str, ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        print(f"  {call.tool_name} {call.args_as_dict()}")
        try:
            reply = input("  Run it? y, or say why not: ").strip()
        except EOFError, KeyboardInterrupt:
            # Ctrl-D, Ctrl-C, or the end of piped input: when in doubt, no.
            print()
            reply = ""
        if reply == "y":
            answers[call.tool_call_id] = ToolApproved()
        else:
            reason = f": {reply}" if reply else "."
            answers[call.tool_call_id] = ToolDenied(
                message=f"Refused at the keyboard{reason}"
            )
    return requests.build_results(approvals=answers)


def approver(*, answer: Answer) -> HandleDeferredToolCalls[None]:
    """A capability that answers the calls held for approval with `answer`."""
    return HandleDeferredToolCalls(handler=answer, id=APPROVER_ID)
