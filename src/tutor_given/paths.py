"""Find the project's folder, so paths don't depend on where a program starts."""

import os
from pathlib import Path


def project_root() -> Path:
    """The folder holding the project's `pyproject.toml`.

    `PYDANTIC_TUTOR_PROJECT_ROOT` overrides the search: a test sets it to run
    in a folder of its own.
    """
    override = os.environ.get("PYDANTIC_TUTOR_PROJECT_ROOT")
    if override:
        return Path(override)
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml above {__file__}")
