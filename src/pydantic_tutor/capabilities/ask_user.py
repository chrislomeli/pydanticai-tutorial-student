"""Putting the model's questions to the person at the keyboard.

The mechanism is the library's: the harness's `AskUser` gives the model a
tool, `ask_user_question`, checks its questions, and sends the answer back
as the tool's return. This module holds what it leaves to the program:
`at_the_menu`, the answerer that shows a question as a numbered menu.
"""

from pydantic_ai_harness.ask_user import (
    AskUserAnswer,
    AskUserRequest,
    AskUserResponse,
    Question,
)


async def at_the_menu(request: AskUserRequest) -> AskUserResponse:
    """Put each of the model's questions to the person, as a numbered menu.

    Enter alone, Ctrl-D or Ctrl-C declines every question in the call: the
    model is told, and makes its own choice.
    """
    answers: list[AskUserAnswer] = []
    for question in request.questions:
        print(f"  {question.header}: {question.question}")
        for number, option in enumerate(iterable=question.options, start=1):
            about = f" ({option.description})" if option.description else ""
            print(f"    {number}. {option.label}{about}")
        numbers = "numbers, with commas" if question.multi_select else "a number"
        reply = printable(
            text=read_line(
                prompt=f"  Pick {numbers}, or type your own answer; Enter declines: "
            )
        )
        if not reply:
            return AskUserResponse(cancelled=True)
        answers.append(read_choice(question=question, reply=reply))
    return AskUserResponse(answers=tuple(answers))


def read_line(*, prompt: str) -> str:
    """A line typed at the keyboard; empty once input has ended."""
    try:
        return input(prompt)
    except EOFError, KeyboardInterrupt:
        # Ctrl-D, Ctrl-C, or the end of piped input.
        print()
        return ""


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
