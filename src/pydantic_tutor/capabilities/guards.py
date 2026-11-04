"""A rule only this application knows, as a hook on the chat's shell commands.

A safety net for mistakes, not a security boundary: `pytest` still runs any code
it is given.
"""

from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# What joins one shell command to another, or runs one inside another.
CHAINING = ("&&", "||", ";", "|", "`", "$(", "\n")


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


guards: Hooks[None] = Hooks(before_tool_execute=refuse_chained_commands)
