"""A capability that caps how many tools the model may call in one response.

A model may ask for many tool calls in one response: asked to read every
module, it can ask for every docstring at once. `BatchLimit` keeps the
first `max_tool_calls` of them, in order, and drops the rest before any
tool runs. The model is sent the results of the calls kept, and can ask
again for the others. It knows nothing about the tools, or about the
run's deps, so any agent can take it.
"""

from dataclasses import dataclass, replace

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, ModelResponsePart, ToolCallPart


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
        # Called after each response, before its tool calls run. What it
        # returns is the response the run goes on with: a dropped call is
        # never run, and the run's history does not hold it.
        kept: list[ModelResponsePart] = []
        calls = 0
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                calls += 1
                if calls > self.max_tool_calls:
                    continue
            kept.append(part)  # text, and the first calls, in their order
        if calls <= self.max_tool_calls:
            return response  # nothing to drop: the response as it came
        return replace(response, parts=kept)
