"""The last run, stopped before it ended, and what the chat does with it.

Given, from the crash lesson (05-05): you run it, you don't write it.
`Unfinished` is what `recovery.py` reads back from the record; `describe` is
the report the chat prints at its start; `resume_point` is where `RESUME` (the
line that carries the run on) starts the run from.
"""

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage

RESUME = "/resume"


@dataclass(frozen=True, kw_only=True)
class Unfinished:
    """The last run, stopped before it ended.

    `finished`: tools whose returns are in `messages`. `unsaved`: tools that
    ran to the end in a round the process didn't finish, so the model may
    ask for them again. `unknown`: tools that started and never ended; their
    work may or may not have been done. `messages`: the history to the last
    settled step, earlier turns included.
    """

    conversation_id: str
    finished: tuple[str, ...]
    unsaved: tuple[str, ...]
    unknown: tuple[str, ...]
    messages: list[ModelMessage]


def describe(*, unfinished: Unfinished) -> str:
    """Say which tools finished and which did not, and what `RESUME` would do."""
    lines = [
        "The last run did not finish: the process stopped during it.",
        f"  Finished: {', '.join(unfinished.finished) or 'none'}",
        f"  Finished, but its return was lost (the model may ask for it again): "
        f"{', '.join(unfinished.unsaved) or 'none'}",
        f"  Started, not finished (may or may not have done its work): "
        f"{', '.join(unfinished.unknown) or 'none'}",
        f"Type {RESUME} to carry it on from its last finished step, "
        "or anything else to leave it.",
    ]
    return "\n".join(lines)


def resume_point(
    *, unfinished: Unfinished, history: list[ModelMessage], revision: int
) -> tuple[list[ModelMessage], int]:
    """The history to carry `unfinished` on from, and the revision to save it at.

    The revision is the chat's if `history` is the same conversation. Otherwise
    it is 0, as for a conversation never saved; if it was saved after all, the
    save is refused as a conflict.
    """
    same = bool(history) and history[-1].conversation_id == unfinished.conversation_id
    return unfinished.messages, revision if same else 0
