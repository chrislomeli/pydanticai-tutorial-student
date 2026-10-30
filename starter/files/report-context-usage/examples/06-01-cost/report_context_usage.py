"""The harness's own gauge, `ReportContextUsage`, and what listening to it changes.

Given, from the cost lesson (06-01). `ReportContextUsage` takes the same
reading as our `ContextGauge`, before each request, and emits it as a
`ContextUsageEvent`, which a listener receives. The same agent runs twice with
`run_sync` on a stand-in model that says whether it was asked in one piece or
streamed: first with `ReportContextUsage` alone, then with a listener for its
events too. Run it from the project root: `uv run examples/06-01-cost/report_context_usage.py`.
"""

from collections.abc import AsyncIterator

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import ContextUsageEvent, ReportContextUsage


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer in one piece, the way a request that isn't streamed is answered."""
    print("  the model was asked: in one piece")
    return ModelResponse(parts=[TextPart(content="Done.")])


async def stream_reply(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    """Answer as a stream, the way a streamed request is answered."""
    print("  the model was asked: streamed")
    yield "Done."


# Both: the run picks one, request by request.
model = FunctionModel(
    function=reply, stream_function=stream_reply, model_name="a stand-in model"
)

listener: Hooks[None] = Hooks()


@listener.on.event(ContextUsageEvent)
async def show_reading(ctx: RunContext[None], event: ContextUsageEvent) -> None:
    """Print each reading as it arrives."""
    print(
        f"  ContextUsageEvent: {event.used_tokens:,} of {event.window_tokens:,} "
        f"tokens, resolved={event.resolved}"
    )


agent: Agent[None, str] = Agent(name="example", output_type=str)

print("ReportContextUsage alone:")
agent.run_sync(user_prompt="Hello.", model=model, capabilities=[ReportContextUsage()])
print("ReportContextUsage, and a listener for its events:")
agent.run_sync(
    user_prompt="Hello.", model=model, capabilities=[ReportContextUsage(), listener]
)
