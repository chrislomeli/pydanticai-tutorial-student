"""Checks that the project from the agent lesson is in place.

Our checks for the agent lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`. They fail until the
lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

from pathlib import Path

import pytest
from pydantic_ai import Agent, models

# This file is tests/01_02_agent/test_agent.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider, so
    # every test here passes only if it needs no key and costs nothing.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


# Three tests moved to tests/01_05_instructions/test_instructions.py: from
# the instructions lesson on, the agent's instructions read the run's
# dependencies, so every run needs `deps=`, which this lesson comes before
# (decisions.md, "A later lesson moves the earlier tests it breaks"). They
# were
# test_the_instructions_travel_on_the_request_not_in_a_part,
# test_inspect_run_shows_the_instructions_apart_from_the_parts and
# test_scripted_run_prints_what_the_function_decided.


def test_the_agent_is_in_the_package_without_a_model() -> None:
    from pydantic_tutor.agent import agent

    assert isinstance(agent, Agent)
    # The model is chosen where the agent is run, not where it is built.
    assert agent.model is None
