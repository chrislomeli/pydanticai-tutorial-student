"""Each chat run recorded step by step, and the last unfinished one read back.

The recording is the harness's: `StepPersistence` writes to a `StepStore`
as the run goes. This module is ours: how the chat's runs are recorded, and
whether the last one ended.
"""

import asyncio

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import StepStore
from pydantic_ai_harness.step_persistence.recovery import inspect_recovery

from tutor_given.unfinished import Unfinished

# A run that ends records one of these. A run with neither stopped without
# ending: the process died, or a Ctrl-C landed while the run's own code ran.
ENDINGS = frozenset({"run_completed", "run_failed"})


def recorder(*, store: StepStore) -> StepPersistence:
    """The capability that records each run it is given, in `store`."""
    # `capture_frontier`: a snapshot before each request too, so a run
    # stopped before its first round of tools finished keeps its prompt.
    return StepPersistence(store=store, capture_frontier=True)


def last_unfinished(*, store: StepStore) -> Unfinished | None:
    """The last run recorded in `store`, if it never ended and has a settled step."""
    return asyncio.run(main=_last_unfinished(store=store))


async def _last_unfinished(*, store: StepStore) -> Unfinished | None:
    runs = await store.list_runs()  # every run, oldest first
    if not runs:
        return None
    last = runs[-1]
    events = await store.list_events(run_id=last.run_id)
    if any(event.kind in ENDINGS for event in events):
        return None
    found = await inspect_recovery(store=store, run_id=last.run_id)
    if found.settled is None:
        return None  # stopped before its first request: nothing to carry on from
    # A round's returns reach a snapshot together, so a call that completed
    # is not always in the settled one.
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
    assert last.conversation_id is not None  # every run has one
    return Unfinished(
        conversation_id=last.conversation_id,
        finished=tuple(dict.fromkeys(name for id_, name in completed if id_ in saved)),
        unsaved=tuple(
            dict.fromkeys(name for id_, name in completed if id_ not in saved)
        ),
        unknown=tuple(effect.tool_name for effect in found.unresolved),
        messages=found.settled.messages,
    )
