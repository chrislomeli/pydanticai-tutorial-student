"""Record what the agent does during a run, one JSON line per hook call.

Milestones:
1. HookName: names a hook the log can record.
2. Entry: one line of the log.
3. Hooks: one per name; each writes an entry and returns what it received.
4. Registration: only the chosen names get a hook.

The hooks return their input unchanged, so the log watches and changes nothing.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import CapabilityOrdering, Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

# The only arguments copied into the log. Named one by one, so a file's
# contents passed to `write_file` never reach it.
LOGGED_ARGUMENTS = ("path", "command")


# 1. HookName
class HookName(StrEnum):
    """The hooks this log can record, named as in `Hooks(...)`."""

    BEFORE_MODEL_REQUEST = "before_model_request"
    BEFORE_TOOL_EXECUTE = "before_tool_execute"
    AFTER_TOOL_EXECUTE = "after_tool_execute"


# 2. Entry
class Entry(BaseModel):
    """One line of the log: names and counts, never contents.

    Fields left as `None` aren't written.
    """

    hook: HookName
    step: int
    tool: str | None = None
    target: str | None = None
    messages: int | None = None
    tools: int | None = None
    result_chars: int | None = None


def build_activity_hooks(
    *, path: Path, hooks: frozenset[HookName] = frozenset(HookName)
) -> Hooks[None]:
    """A capability that logs the chosen hooks to the file at `path`."""

    def write(*, entry: Entry) -> None:
        with path.open(mode="a") as log:
            log.write(entry.model_dump_json(exclude_none=True) + "\n")

    # 3. Hooks
    # `async`, with no `await` inside: when the model asks for several tools
    # at once, these hooks can't write over each other.
    async def before_model_request(
        ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        write(
            entry=Entry(
                hook=HookName.BEFORE_MODEL_REQUEST,
                step=ctx.run_step,
                messages=len(request_context.messages),
                tools=len(request_context.model_request_parameters.function_tools),
            )
        )
        return request_context

    async def before_tool_execute(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        targets = [str(args[name]) for name in LOGGED_ARGUMENTS if name in args]
        write(
            entry=Entry(
                hook=HookName.BEFORE_TOOL_EXECUTE,
                step=ctx.run_step,
                tool=call.tool_name,
                target=targets[0] if targets else None,
            )
        )
        return args

    async def after_tool_execute(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: object,
    ) -> object:
        write(
            entry=Entry(
                hook=HookName.AFTER_TOOL_EXECUTE,
                step=ctx.run_step,
                tool=call.tool_name,
                result_chars=len(str(result)),
            )
        )
        return result

    # 4. Registration
    return Hooks[None](
        before_model_request=(
            before_model_request if HookName.BEFORE_MODEL_REQUEST in hooks else None
        ),
        before_tool_execute=(
            before_tool_execute if HookName.BEFORE_TOOL_EXECUTE in hooks else None
        ),
        after_tool_execute=(
            after_tool_execute if HookName.AFTER_TOOL_EXECUTE in hooks else None
        ),
        # Outermost, wherever it is listed: its `before_` hooks see every
        # attempt, and its `after_` hooks see the result the model gets.
        ordering=CapabilityOrdering(position="outermost"),
    )
