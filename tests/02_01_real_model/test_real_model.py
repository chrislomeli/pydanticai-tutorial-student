"""Checks that the project from the real-model lesson is in place.

Our checks for the real-model lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them calls a real model.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from pathlib import Path

# This file is tests/02_01_real_model/test_real_model.py: two folders below
# the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def test_the_model_is_one_setting() -> None:
    from pydantic_tutor.model import CHAT_MODEL

    assert CHAT_MODEL == "openai:gpt-4o-mini"


def test_without_a_key_the_run_says_which_key_it_needs() -> None:
    # A real run needs the key; with none, the script must fail before
    # anything is sent, naming the variable to set.
    env = {
        name: value for name, value in os.environ.items() if name != "OPENAI_API_KEY"
    }
    result = subprocess.run(
        [sys.executable, "scripts/first_real_run.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**env, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr
