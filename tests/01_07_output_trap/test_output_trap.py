"""Checks that the project from the output-trap lesson is in place.

Our checks for the output-trap lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

The lesson changes only a script, so these run it, and ask mypy what it
makes of the agent the script builds.
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

    lines = printed.splitlines()
    extractor_part = lines[: lines.index('The desk, told "Hello":')]
    # Offered no text answer, twice: the first answer and the retry.
    assert extractor_part.count("  a text answer offered: False") == 2
    assert "Invalid JSON" in printed
    assert extractor_part[-1] == (
        "  the run failed: Exceeded maximum output retries (1)"
    )


def test_the_desk_answers_in_words_or_with_an_invoice() -> None:
    printed = run_script()

    lines = printed.splitlines()
    assert lines[lines.index('The desk, told "Hello":') :] == [
        'The desk, told "Hello":',
        "  a text answer offered: True",
        "  words: Hello! Send me an invoice and I will read it.",
        "The desk, sent INV-0042:",
        "  a text answer offered: True",
        "  an Invoice: INV-0042 85.50",
    ]


def test_mypy_knows_the_desk_may_answer_with_either(tmp_path: Path) -> None:
    # The lesson's point about types: without the type parameters mypy
    # takes the desk's output to be `str` alone. Ask it, on a copy of the
    # script, what type the desk has.
    check = tmp_path / "check.py"
    check.write_text(SCRIPT.read_text() + "\nreveal_type(desk)\n")
    result = subprocess.run(
        args=[
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(PROJECT_ROOT / "pyproject.toml"),
            "--no-incremental",
            str(check),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MYPYPATH": str(PROJECT_ROOT / "src")},
    )
    assert (
        'Revealed type is "pydantic_ai.agent.Agent[None, str | '
        'pydantic_tutor.invoices.Invoice]"' in result.stdout
    ), result.stdout
    assert "Success: no issues found" in result.stdout, result.stdout
