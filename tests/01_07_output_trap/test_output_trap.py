"""Checks that the project from the output-trap lesson is in place.

Our checks for the output-trap lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

The lesson changes only a script, so these run it.
"""

import os
import subprocess
import sys
from pathlib import Path

# This file is tests/01_07_output_trap/test_output_trap.py: two folders
# below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPT: Path = PROJECT_ROOT / "scripts" / "output_trap.py"


def run_script() -> str:
    """Run the lesson's script from the project root; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"output_trap.py failed:\n{result.stderr}"
    return result.stdout


def test_hello_ends_the_extractors_run_after_a_retry() -> None:
    printed = run_script()

    extractor_part = printed.splitlines()
    # Offered no text answer, twice: the first answer and the retry.
    assert extractor_part.count("  a text answer offered: False") == 2
    assert "Invalid JSON" in printed
    assert extractor_part[-1] == (
        "  the run failed: Exceeded maximum output retries (1)"
    )
