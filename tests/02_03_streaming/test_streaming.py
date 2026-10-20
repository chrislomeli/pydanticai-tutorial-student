"""Checks that the project from the Streaming lesson is in place.

Our checks for the Streaming lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import models

# This file is tests/02_03_streaming/test_streaming.py: two folders below
# the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_the_answer_streams_in_more_than_one_growing_chunk() -> None:
    # Goal 1: stream_text() yields the answer as it is written, not as one
    # blob — each chunk is a growing prefix of the final answer.
    printed = run_script("scripts/stream_run.py")
    section = printed.split("=== streaming the answer ===")[1].split(
        "=== walking the run ==="
    )[0]
    chunks = [
        line.removeprefix("chunk: ").strip("'")
        for line in section.splitlines()
        if line.startswith("chunk: ")
    ]

    assert len(chunks) >= 2  # more than one chunk: it streamed
    assert chunks[0] != chunks[-1]  # the first chunk is not the whole answer
    for earlier, later in itertools.pairwise(chunks):
        assert later.startswith(earlier)  # each chunk is cumulative

    answer_line = next(
        line for line in section.splitlines() if line.startswith("answer: ")
    )
    assert answer_line == f"answer: '{chunks[-1]}'"


def test_each_node_s_events_come_from_that_node() -> None:
    # Goal 2: walking the run with agent.iter() visits a ModelRequestNode
    # before CallToolsNode, in order; the model-request node's own events
    # are the model's part events, the call-tools node's own events are the
    # tool-call events — never mixed.
    printed = run_script("scripts/stream_run.py")
    section = printed.split("=== walking the run ===")[1]
    lines = [line for line in section.splitlines() if line.strip()]

    node_lines = [
        line
        for line in lines
        if not line.startswith(" ") and not line.startswith("output: ")
    ]
    assert node_lines == [
        "UserPromptNode",
        "ModelRequestNode",
        "CallToolsNode",
        "ModelRequestNode",
        "CallToolsNode",
        "End",
    ]

    # Group each node's own event lines under it, in the order printed.
    events_by_node: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("output: "):
            break
        if not line.startswith(" "):
            events_by_node.append((line, []))
        else:
            events_by_node[-1][1].append(line.strip())

    model_request_events = {
        e
        for node, events in events_by_node
        if node == "ModelRequestNode"
        for e in events
    }
    call_tools_events = {
        e for node, events in events_by_node if node == "CallToolsNode" for e in events
    }

    assert model_request_events & call_tools_events == set()  # never mixed
    assert all(e.startswith("[model]") for e in model_request_events)
    assert all(e.startswith("[tool]") for e in call_tools_events)
    assert "[model] PartDeltaEvent" in model_request_events  # the streamed text
    assert "[tool] FunctionToolCallEvent" in call_tools_events  # the tool call
    assert "[tool] FunctionToolResultEvent" in call_tools_events  # the tool's result

    output_line = next(
        line for line in section.splitlines() if line.startswith("output: ")
    )
    assert output_line == "output: 'The jobs module manages a queue of jobs.'"
