"""A capability that caps how many tool calls of one model response run.

From the capability lesson (04-03). `BatchLimit` keeps the first
`max_tool_calls` of a response's tool calls and drops the rest before any
runs. The model is sent the results of the calls kept, and can ask again
for the others. It reads no deps, so any agent can take it.
"""

from dataclasses import dataclass, replace

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, ToolCallPart


@dataclass(kw_only=True)
class BatchLimit[DepsT](AbstractCapability[DepsT]):
    """Keep at most `max_tool_calls` tool calls from each model response."""

    # How many tool calls of one response run. The rest are dropped.
    max_tool_calls: int

    def __post_init__(self) -> None:
        """Refuse a cap that would drop every call."""
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")

    async def after_model_request(
        self,
        ctx: RunContext[DepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Return the response without its tool calls past the cap."""
        # What this returns is the response the run goes on with: a dropped
        # call is never run, and the history does not hold it.
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if len(calls) <= self.max_tool_calls:
            return response  # nothing to drop: the response as it came
        dropped = calls[self.max_tool_calls :]
        # Text, and the first calls, stay in the order the model gave them.
        kept = [part for part in response.parts if all(part is not d for d in dropped)]
        return replace(response, parts=kept)
