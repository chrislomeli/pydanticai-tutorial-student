"""Checks that the project from the setup lesson is in place.

Our checks for the setup lesson: they verify the lesson, and the student is
not asked to run them. The project has no pytest of its own, so run them
with `uv run --with pytest pytest`. They fail until the lesson is done.

Every import of the project or the library is inside the test that needs it,
so that before the install each test fails on its own, instead of the whole
file failing to load.
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

# This file is tests/01_01_setup/test_setup.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def test_python_is_3_14() -> None:
    assert sys.version_info[:2] == (3, 14), (
        f"running on Python {sys.version_info.major}.{sys.version_info.minor}; "
        "the project should pin 3.14 in .python-version"
    )


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        ("pydantic-ai", "2.56.0"),
        # Not added by you: pydantic-ai and pydantic-ai-harness both require
        # exactly this version, so the two pins move together.
        ("pydantic-ai-slim", "2.56.0"),
        ("pydantic-ai-harness", "0.56.0"),
    ],
)
def test_versions_are_pinned(package: str, expected: str) -> None:
    installed: str | None
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None  # not installed: the assert below reports None
    assert installed == expected, f"{package}: installed {installed}"


def test_package_is_installed_from_src() -> None:
    import pydantic_tutor

    # In a src/ layout the package can only be imported once it is installed,
    # so this proves the install, not just that the folder exists.
    package_dir = Path(pydantic_tutor.__file__).resolve().parent
    assert package_dir == PROJECT_ROOT / "src" / "pydantic_tutor"


def test_env_is_ignored() -> None:
    # Holds whether or not .env exists: it proves the file can never be
    # committed. git check-ignore exits 0 when the path is ignored.
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".env"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0, ".env is not ignored by .gitignore"


def test_agent_runs_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai import Agent, models
    from pydantic_ai.models.test import TestModel

    # Any request to a real model now raises instead of reaching a provider,
    # so this test passes only if it needs no key and costs nothing.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)

    result = Agent(TestModel()).run_sync("hello")

    assert result.output == "success (no tool calls)"
