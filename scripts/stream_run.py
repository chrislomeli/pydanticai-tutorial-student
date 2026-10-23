"""Stream an answer as it is written, then walk the same run node by node.

From the streaming lesson (02-03). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps

PROMPT = "What does the jobs module do, in one short sentence?"
ANSWER_WORDS = ["The ", "jobs ", "module ", "manages ", "a ", "queue ", "of ", "jobs."]


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model that reads the tool's result, then answers.

    Called once per request: the first carries the user's prompt, the
    second carries the tool's result.
    """
    request = messages[-1]
    for part in request.parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content="".join(ANSWER_WORDS))])
    call = ToolCallPart(tool_name="read_docstring", args={"module": "tinyjobs.jobs"})
    return ModelResponse(parts=[call])


async def stream_reply(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """The streaming twin of `reply`: yields the same answer, word by word.

    The 0.15s pause between words is wider than `stream_text()`'s own
    0.1s debounce window, so each word arrives as its own chunk instead
    of being coalesced with the next.
    """
    request = messages[-1]
    for part in request.parts:
        if isinstance(part, ToolReturnPart):
            for word in ANSWER_WORDS:
                await asyncio.sleep(0.15)
                yield word
            return
    yield {
        1: DeltaToolCall(name="read_docstring", json_args='{"module": "tinyjobs.jobs"}')
    }


def deps() -> CodebaseDeps:
    return CodebaseDeps(root=Path("sample_code/tinyjobs"))


async def stream_the_answer() -> None:
    """Part 1: stream the final answer as it is written."""
    print("=== streaming the answer ===")
    model = FunctionModel(function=reply, stream_function=stream_reply)
    async with agent.run_stream(user_prompt=PROMPT, model=model, deps=deps()) as stream:
        async for chunk in stream.stream_text():
            print(f"chunk: {chunk!r}")
        print(f"answer: {await stream.get_output()!r}")


async def walk_the_run() -> None:
    """Part 2: the same turn, walked node by node with agent.iter()."""
    print("=== walking the run ===")
    model = FunctionModel(function=reply, stream_function=stream_reply)
    async with agent.iter(user_prompt=PROMPT, model=model, deps=deps()) as run:
        async for node in run:
            print(type(node).__name__)
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        print(f"  [model] {type(event).__name__}")
            elif agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        print(f"  [tool] {type(event).__name__}")
        assert run.result is not None
        print(f"output: {run.result.output!r}")


async def main() -> None:
    await stream_the_answer()
    await walk_the_run()


if __name__ == "__main__":
    asyncio.run(main())
