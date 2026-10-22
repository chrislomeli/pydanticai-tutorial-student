"""Each chat run recorded step by step, so a run a crash stopped can be carried on.

The recording is the harness's: `StepPersistence`, given to a run, writes
to a `SqliteStepStore` as the run goes. This module is ours: it says how
the chat's runs are recorded, and reads the record back to answer one
question when the chat starts: did the last run end?
"""

import asyncio
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import SqliteStepStore
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery

# A run that ends records one of these: it completed, or it failed (an
# error, a limit, almost every Ctrl-C). A run with neither stopped without
# ending: the process died, or a Ctrl-C landed while the run's own code
# was running, which the library lets through without recording it.
ENDINGS = frozenset({"run_completed", "run_failed"})


@dataclass(frozen=True, kw_only=True)
class Unfinished:
    """The last run, stopped before it ended: what is known about it."""

    conversation_id: str
    # Tools that ran to the end, with their returns in `messages`: a run
    # carrying it on starts with them done.
    finished: tuple[str, ...]
    # Tools that ran to the end in a round the process didn't finish: their
    # work was done, but their returns are not in `messages`, so the model
    # may ask for them again.
    unsaved: tuple[str, ...]
    # Tools that started and never finished: their work may or may not
    # have been done.
    unknown: tuple[str, ...]
    # The run's history up to its last settled step, every earlier turn
    # included: what a run carrying it on starts from.
    messages: list[ModelMessage]


def recorder(*, store: SqliteStepStore) -> StepPersistence:
    """The capability that records each run it is given, in `store`."""
    # `capture_frontier`: a snapshot before each request too, so a run
    # stopped before its first round of tool calls finished still keeps
    # its prompt.
    return StepPersistence(store=store, capture_frontier=True)


def last_unfinished(*, store: SqliteStepStore) -> Unfinished | None:
    """The last run recorded in `store`, if it never ended and has a settled step."""
    return asyncio.run(main=_last_unfinished(store=store))


async def _last_unfinished(*, store: SqliteStepStore) -> Unfinished | None:
    runs = await store.list_runs()  # every run, oldest first
    if not runs:
        return None
    last = runs[-1]
    events = await store.list_events(run_id=last.run_id)
    if any(event.kind in ENDINGS for event in events):
        return None
    found = await inspect_recovery(store=store, run_id=last.run_id)
    if found.settled is None:
        # Stopped before its first request was sent: no tool had started,
        # and there is nothing to carry on from.
        return None
    # A round's returns reach a snapshot together, once every call in it
    # has ended, so a call that completed is not always in the settled one.
    saved = {
        part.tool_call_id
        for message in found.settled.messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    completed = [
        (event.tool_call_id, event.tool_name)
        for event in events
        if event.kind == "tool_call_completed" and event.tool_name is not None
    ]
    # Every run has a conversation id: the one it was given, or a new one.
    assert last.conversation_id is not None
    return Unfinished(
        conversation_id=last.conversation_id,
        # Each name once, in the order the tools completed.
        finished=tuple(dict.fromkeys(name for id_, name in completed if id_ in saved)),
        unsaved=tuple(
            dict.fromkeys(name for id_, name in completed if id_ not in saved)
        ),
        unknown=tuple(effect.tool_name for effect in found.unresolved),
        messages=found.settled.messages,
    )
