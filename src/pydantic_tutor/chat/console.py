"""Terminal colors for the chat, used only where someone will see them.

From the console-chat lesson (02-02). Text is colored only when the output
is a terminal and `NO_COLOR` is not set, so piped output, files and tests
get the plain text.
"""

import os
import sys

_BLUE = "\033[38;5;75m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _wanted() -> bool:
    """Whether to color: output goes to a terminal, and `NO_COLOR` is unset."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def blue(*, text: str) -> str:
    """The text in blue, or as it is when colors are not wanted."""
    return f"{_BLUE}{text}{_RESET}" if _wanted() else text


def dim(*, text: str) -> str:
    """The text dimmed, or as it is when colors are not wanted."""
    return f"{_DIM}{text}{_RESET}" if _wanted() else text
