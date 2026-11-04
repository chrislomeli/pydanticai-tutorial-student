"""Checks that the project from the limits lesson is in place.

Our checks for the limits lesson: they verify the lesson, and the student
is not asked to run them. Run them with `uv run pytest`. They fail until
the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelRetry, ToolFailed, models

# This file is tests/01_08_limits/test_limits.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        [sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_a_misspelling_is_retried_and_a_feed_that_is_down_fails() -> None:
    from pydantic_tutor.stock import stock_of

    assert stock_of(product="apple") == 12
    with pytest.raises(ModelRetry):
        stock_of(product="bred")
    with pytest.raises(ToolFailed):
        stock_of(product="bread")


def test_the_run_goes_on_after_both() -> None:
    printed = run_script("scripts/stock_check.py")

    assert "asked to retry: There is no product 'bred'." in printed
    assert (
        'tool return (failed): {"error":"The stock feed for bread is down."}' in printed
    )
    assert printed.strip().splitlines()[-1] == "output: I can't check bread right now."


def test_a_request_limit_stops_a_runaway() -> None:
    printed = run_script("scripts/runaway.py")

    lines = printed.strip().splitlines()
    assert lines[:-1] == [f"request {n}: the model asks for bread" for n in range(1, 6)]
    assert lines[-1].startswith(
        "stopped: The next request would exceed the request_limit of 5."
    )


def test_a_slow_tool_times_out() -> None:
    printed = run_script("scripts/slow_tool.py")

    assert "asked to retry: Timed out after 0.5 seconds." in printed
    assert printed.strip().splitlines()[-1] == "output: The supplier didn't answer."
