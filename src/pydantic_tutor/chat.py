"""The chat: read a line, run the agent on it, print the answer; you answer each held call.

In the chat, the model can also ask you a question, as a menu you answer
at the keyboard. `chat(app=...)` returns when you type /exit or input
ends. A turn that fails ends that turn, never the chat.
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
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import (
    AskUserAnswer,
    AskUserRequest,
    AskUserResponse,
    Question,
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


async def answer_at_keyboard(request: AskUserRequest) -> AskUserResponse:
    """Put each of the model's questions to the person, as a numbered menu."""
    answers: list[AskUserAnswer] = []
    for question in request.questions:
        print(f"  {question.header}: {question.question}")
        for number, option in enumerate(iterable=question.options, start=1):
            about = f" ({option.description})" if option.description else ""
            print(f"    {number}. {option.label}{about}")
        numbers = "numbers, with commas" if question.multi_select else "a number"
        typed = ask(
            prompt=f"  Pick {numbers}, or type your own answer; Enter declines: "
        )
        reply = printable(text=typed or "")
        if not reply:
            # Enter alone, Ctrl-D or Ctrl-C declines every question in the
            # call: the model is told, and makes its own choice.
            return AskUserResponse(cancelled=True)
        answers.append(read_choice(question=question, reply=reply))
    return AskUserResponse(answers=tuple(answers))


def printable(*, text: str) -> str:
    """`text` with each control character as a space, stripped.

    `AskUser` fails the whole run on an answer with a control character in
    it, such as a tab or an arrow key's escape code: nothing a person types
    may cost them the turn.
    """
    return "".join(char if char.isprintable() else " " for char in text).strip()


def read_choice(*, question: Question, reply: str) -> AskUserAnswer:
    """The options a reply's numbers pick; any other reply is the person's own answer."""
    labels = {
        str(number): option.label
        for number, option in enumerate(iterable=question.options, start=1)
    }
    parts = reply.split(sep=",") if question.multi_select else [reply]
    picks = [part.strip() for part in parts]
    if all(pick in labels for pick in picks):
        # A number typed twice picks its option once, in the order typed.
        selected = tuple(dict.fromkeys(labels[pick] for pick in picks))
        return AskUserAnswer(header=question.header, selected=selected)
    return AskUserAnswer(header=question.header, custom_answer=reply)


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
    # agent's refusal (the same `id`); the model may ask you a question
    # (`AskUser` belongs here, not on the agent: only the chat's runs have
    # someone to answer); and each tool that ran is named.
    capabilities: list[AgentCapability[RepoDeps]] = [
        approver(answer=ask_at_keyboard),
        AskUser(answerer=answer_at_keyboard),
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
