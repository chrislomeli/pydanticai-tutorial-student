"""Logging setup for the command line.

Log records go to stderr, so the command's own output on stdout stays
pipeable.
"""

from __future__ import annotations

import logging
import sys


def configure(level: int = logging.WARNING) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("tinyjobs")
    root.handlers[:] = [handler]
    root.setLevel(level)
