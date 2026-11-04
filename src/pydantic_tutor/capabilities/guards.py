"""Two rules only this application knows, as hooks on the chat's tool calls.

A safety net for mistakes, not a security boundary: `pytest` still runs any code
it is given, and `TOKEN` matches one known format, not every secret.
"""

import re
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# What joins one shell command to another, or runs one inside another.
CHAINING = ("&&", "||", ";", "|", "`", "$(", "\n")
# The shape of this project's tokens.
TOKEN = re.compile(r"tj-[A-Za-z0-9-]+")


async def refuse_chained_commands(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Refuse a shell command that holds more than one command."""
    command = str(args.get("command", ""))
    if any(mark in command for mark in CHAINING):
        raise ModelRetry("Run one command at a time, with no chaining.")
    return args


async def remove_tokens(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: object,
) -> object:
    """Return the tool's result with this project's tokens removed."""
    if isinstance(result, str):
        return TOKEN.sub("[token removed]", result)
    return result


guards: Hooks[None] = Hooks(
    before_tool_execute=refuse_chained_commands,
    after_tool_execute=remove_tokens,
)
