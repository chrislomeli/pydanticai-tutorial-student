"""A chat in the terminal that remembers: type a message, read the answer.

From the console-chat lesson (02-02); history from the history lesson
(02-03); streaming from the streaming lesson (02-04); its agent and model
from the settings lesson (03-01); you answer the writes and commands the
agent holds from the chat-loop lesson (05-02). Run it with `OPENAI_API_KEY` set:
`uv run python -m pydantic_tutor`. An empty line, or Ctrl-D, ends
it. Each run is handed the messages of the runs before it, so the model is
sent the whole chat so far, and its answer is printed as it is written. When
the agent wants to write a file or run a command, it shows the call and waits:
`y` runs it, anything else refuses it. The model may also ask you a question,
from the ask-user lesson (05-03): a numbered menu; Enter declines. After each
answer, a status line from the cost lesson (06-01) says how full the last
request's context was, and how many tokens the turn sent in all.
"""

from pydantic import ValidationError
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness import AskUser

from pydantic_tutor.capabilities.approval import approver, at_the_keyboard
from pydantic_tutor.capabilities.ask_user import at_the_menu
from pydantic_tutor.capabilities.context_gauge import ContextGauge
from pydantic_tutor.capabilities.recovery import last_unfinished, recorder
from pydantic_tutor.chat.console import blue, dim
from pydantic_tutor.composition import App, build_app
from pydantic_tutor.settings import Settings
from pydantic_tutor.stores.conversations import ConversationConflictError
from tutor_given.unfinished import RESUME, Unfinished, describe, resume_point

NEW = "/new"


def main(*, app: App | None = None) -> None:
    """Read a line, run the agent on it with the chat so far, print the answer.

    The chat runs `app`; given none, it builds the app from the settings.
    """
    if app is None:
        try:
            app = build_app(settings=Settings())
        except (ValidationError, UserError) as error:
            # A bad setting or a missing key: say which, and do not open the chat.
            raise SystemExit(f"pydantic-tutor cannot start:\n{error}") from error
    print("Chat with the agent. An empty line ends it.")
    history: list[ModelMessage] = []  # every message of the chat so far
    # The revision of `history` in the store; 0: not saved yet.
    revision = 0
    store = app.store
    saved = None if store is None else store.latest()
    if saved is not None:
        history, revision = saved.messages, saved.revision
        print(
            f"Carrying on conversation {history[-1].conversation_id}: "
            f"{len(history)} messages. Type {NEW} to start another."
        )
    # Given to every run, because only the chat has someone to answer: you
    # answer the calls it holds, in place of the agent's own answerer (the
    # same `id`), and the model may ask you a question. The agent has no
    # `AskUser`: a run with no one there would only be told "declined".
    at_the_desk: list[AgentCapability[None]] = [
        approver(answer=at_the_keyboard),
        AskUser(answerer=at_the_menu),
    ]
    unfinished: Unfinished | None = None
    if app.steps is not None:
        # Each run is recorded as it goes, so a crash leaves a record.
        at_the_desk.append(recorder(store=app.steps))
        unfinished = last_unfinished(store=app.steps)
        if unfinished is not None:
            print(describe(unfinished=unfinished))
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:  # Ctrl-D, or the end of piped input
            break
        if not line:
            break
        prompt: str | None = line
        start, start_revision = history, revision
        if line == RESUME:
            if unfinished is None:
                print("There is no unfinished run to carry on.")
                continue
            # No prompt: the run picks up where the history stops, and the
            # model is asked what to do next.
            prompt = None
            start, start_revision = resume_point(
                unfinished=unfinished, history=history, revision=revision
            )
        # The offer lasts one line: the next run becomes the last recorded.
        unfinished = None
        if line == NEW:
            history, revision = [], 0  # the saved one stays saved
            print("A new conversation starts with your next line.")
            continue
        # A new gauge for each turn: it reads every request this turn sends,
        # and only those.
        gauge = ContextGauge()
        result = app.agent.run_stream_sync(
            user_prompt=prompt,
            model=app.model,
            message_history=start,
            capabilities=[*at_the_desk, gauge],
        )
        print(blue(text="agent> "), end="")
        # Each piece of the answer as it arrives; flush, so it shows at once.
        for piece in result.stream_text(delta=True):
            print(blue(text=piece), end="", flush=True)
        print()
        print(dim(text=f"  {gauge.status_line()}"))
        if store is not None:
            try:
                revision = store.save(
                    messages=result.all_messages(), revision=start_revision
                )
            except ConversationConflictError as error:
                # Keep `history` as it was: what the chat carries is what is saved.
                print(f"This turn was not kept: {error}. Type {NEW}, or restart.")
                continue
        history = result.all_messages()
