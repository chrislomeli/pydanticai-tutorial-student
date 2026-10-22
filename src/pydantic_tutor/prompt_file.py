"""A capability that gives an agent its instructions from a text file.

An agent's prompt is text, not code: in a file of its own, it can be read,
reviewed and changed without touching Python. `PromptFile` reads the file
once, when it is made, and supplies the text as the agent's instructions.
It knows nothing about the run's deps, so any agent can take it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering


@dataclass(kw_only=True)
class PromptFile[DepsT](AbstractCapability[DepsT]):
    """Supply the text of a file as the agent's instructions."""

    # The file the prompt is read from, as UTF-8 text.
    path: Path
    # The file's text, without the blank lines and spaces around it.
    text: str = field(init=False)

    def __post_init__(self) -> None:
        """Read the prompt now, so a missing or empty file fails at once."""
        self.text = self.path.read_text(encoding="utf-8").strip()
        if not self.text:
            raise ValueError(f"the prompt file {self.path} is empty")

    def get_instructions(self) -> str:
        """Return the prompt, as text."""
        # Text, not a function: the agent sends text it was given before
        # any it computes for a request, and does not ask for it again.
        # An edit to the file is read when the program next starts.
        return self.text

    def get_ordering(self) -> CapabilityOrdering:
        """Put this capability first, wherever it is listed."""
        # The prompt says what the agent is for, so it is the first text the
        # model is sent: of the instructions given as text, the agent sends
        # those of the outermost capability first.
        return CapabilityOrdering(position="outermost")
