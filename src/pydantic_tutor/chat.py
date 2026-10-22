"""The chat: read a line, run the agent on it, print the answer; you answer each held call.

`chat(app=...)` returns when you type /exit or input ends. A turn that
fails ends that turn, never the chat.
"""

import traceback

import pydantic_ai
from pydantic_ai import Conversation, RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)

from pydantic_tutor.approval import approver
from pydantic_tutor.composition import App
from pydantic_tutor.repo_tools import RepoDeps

PROMPT = "pydantic-tutor ➤ "
EXIT = "/exit"


def ask(*, prompt: str) -> str | None:
    """A line typed at the keyboard, stripped; None once input has ended."""
    try:
        return input(prompt).strip()
    except EOFError, KeyboardInterrupt:
        # Ctrl-D, Ctrl-C, or the end of piped input.
        print()
        return None


def ask_at_keyboard(
    ctx: RunContext[RepoDeps], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Show each call held for approval; run it only if the person types y."""
    answers: dict[str, ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        print(f"  {call.tool_name} {call.args_as_dict()}")
        reply = ask(prompt="  Run it? y, or say why not: ")
        if reply == "y":
            answers[call.tool_call_id] = ToolApproved()
        else:
            # Anything else refuses, nothing typed included: the model is
            # told, with the person's reason if they gave one.
            reason = f": {reply}" if reply else "."
            answers[call.tool_call_id] = ToolDenied(
                message=f"Refused at the keyboard{reason}"
            )
    return requests.build_results(approvals=answers)


async def show_tool_call(
    ctx: RunContext[RepoDeps],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, object],
    result: object,
) -> object:
    """Name each tool that ran, and pass its result on unchanged."""
    print(f"  ran {call.tool_name}")
    return result


def chat(*, app: App) -> None:
    """Chat with the agent until /exit: each line typed is one run."""
    # The chat owns this terminal: no first-run banner in its first answer.
    pydantic_ai.BANNER_ENABLED = False
    print(f"Chatting with {app.model.model_name}. Type {EXIT} to leave.")
    # Given to every run: you answer its held calls, in place of the
    # agent's refusal (the same `id`), and each tool that ran is named.
    capabilities: list[AgentCapability[RepoDeps]] = [
        approver(answer=ask_at_keyboard),
        Hooks(after_tool_execute=show_tool_call),
    ]
    conversation: Conversation | None = None
    while True:
        text = ask(prompt=PROMPT)
        if text is None or text == EXIT:
            return
        if not text:
            continue
        # A turn is what fails, not the chat: whatever stops one (the
        # provider, a limit, a bug in a tool, Ctrl-C), the conversation
        # stays as it was before it, and the next line starts from there.
        try:
            result = app.agent.run_sync(
                user_prompt=text,
                model=app.model,
                deps=app.deps,
                conversation=conversation,
                capabilities=capabilities,
            )
        except KeyboardInterrupt:
            print("Interrupted.")
        except AgentRunError as error:
            # A failure the library reports: one line says what it was.
            print(f"{type(error).__name__}: {error}")
        except Exception:
            # Anything else is a bug, in a tool or below it: the traceback
            # says where, for whoever fixes it.
            traceback.print_exc()
        else:
            conversation = result.conversation
            print(result.output)
