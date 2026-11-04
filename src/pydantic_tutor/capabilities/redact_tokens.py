"""Remove tokens from tool results, and tell the model that it happens.

A safety net for mistakes, not a security boundary: it matches one known
format, not every secret.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

REPLACEMENT = "[token removed]"


@dataclass(kw_only=True)
class RedactTokens[DepsT](AbstractCapability[DepsT]):
    """Replace anything matching `pattern` in tool results, and say so."""

    # A regular expression for the tokens to remove.
    pattern: str
    compiled: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        """Compile the pattern now, so a bad one fails when the app starts."""
        self.compiled = re.compile(self.pattern)

    def get_instructions(self) -> str:
        """Tell the model what the replacement text means."""
        return (
            f"{REPLACEMENT} in a tool result marks a secret removed on purpose. "
            "The file isn't broken. Don't try to recover the secret."
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[DepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: object,
    ) -> object:
        """Return the tool's result with the tokens removed."""
        if isinstance(result, str):
            return self.compiled.sub(REPLACEMENT, result)
        return result
