"""A stand-in model that says what each request carried, and counts it as a provider would.

Given, from the cost lesson (06-01): you run it, you don't write it. The model
lists the modules, reads one docstring, then answers: three requests a turn.
For each request it prints the tokens of the request's three parts, the
instructions, the tool definitions and the messages, at about four characters
a token. It reports their sum as the request's input tokens, as a provider
reports what it counted. A `FunctionModel` does not do that itself: the count
it reports is the text of the messages alone.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

# The usual rule of thumb for English text and code; a real model's
# tokenizer counts its own way.
CHARS_PER_TOKEN = 4


def message_text(*, message: ModelMessage) -> str:
    """The text of one message that a provider is sent, near enough."""
    texts: list[str] = []
    for part in message.parts:
        if isinstance(part, UserPromptPart | TextPart):
            texts.append(str(part.content))
        elif isinstance(part, ToolCallPart):
            texts.append(part.tool_name + part.args_as_json_str())
        elif isinstance(part, ToolReturnPart):
            texts.append(part.model_response_str())
    return "".join(texts)


def definition_text(*, tool: ToolDefinition) -> str:
    """A tool definition as text: its name, description and argument schema."""
    schema = json.dumps(obj=tool.parameters_json_schema)
    return tool.name + (tool.description or "") + schema


class ProviderCounts(FunctionModel):
    """A streaming `FunctionModel` whose responses report the input tokens it was sent.

    The stream function leaves the count in `sent`; each response reports it.
    """

    sent: int = 0

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            # The stream function has run by now, up to its first item. The
            # response's usage is where a provider's count goes; there is no
            # public way to set it on a streamed response. (A streamed
            # `FunctionModel` response starts at 50 input tokens, an
            # allowance for nothing; this replaces it.)
            response._usage.input_tokens = self.sent
            yield response


def counts_like_a_provider() -> ProviderCounts:
    """A model that lists the modules, reads a docstring, answers, and prints what each request carried.

    From the cost lesson (06-01); it streams, so the chat can use it. It reads
    the turn so far, from its last prompt on, and asks for the first of its
    two tools that has not returned in it yet.
    """

    def split(*, messages: list[ModelMessage], info: AgentInfo) -> tuple[int, int, int]:
        """The tokens of the instructions, the tool definitions and the messages."""
        instructions = len(info.instructions or "") // CHARS_PER_TOKEN
        tools = (
            sum(len(definition_text(tool=tool)) for tool in info.function_tools)
            // CHARS_PER_TOKEN
        )
        history = (
            sum(len(message_text(message=message)) for message in messages)
            // CHARS_PER_TOKEN
        )
        return instructions, tools, history

    def next_call(*, messages: list[ModelMessage]) -> tuple[str, str] | None:
        """The next tool call as (name, arguments as JSON), or None once both returned."""
        start = max(
            index
            for index, message in enumerate(messages)
            if any(isinstance(part, UserPromptPart) for part in message.parts)
        )
        returned = {
            part.tool_name
            for message in messages[start:]
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        if "list_modules" not in returned:
            return "list_modules", "{}"
        if "read_docstring" not in returned:
            return "read_docstring", json.dumps({"module": "tinyjobs.jobs"})
        return None

    def answer(*, messages: list[ModelMessage]) -> str:
        """The first line of the docstring that came back in this turn."""
        start = max(
            index
            for index, message in enumerate(messages)
            if any(isinstance(part, UserPromptPart) for part in message.parts)
        )
        docstring = next(
            part.model_response_str()
            for message in messages[start:]
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "read_docstring"
        )
        return f"tinyjobs.jobs: {docstring.splitlines()[0]}"

    def report(*, messages: list[ModelMessage], info: AgentInfo) -> int:
        """Print what this request carries; return its tokens, as a provider counts them."""
        instructions, tools, history = split(messages=messages, info=info)
        number = 1 + sum(isinstance(message, ModelResponse) for message in messages)
        print(
            f"  request {number}: instructions {instructions:,} · "
            f"tool definitions {tools:,} ({len(info.function_tools)}) · "
            f"messages {history:,} ({len(messages)})"
        )
        return instructions + tools + history

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Not used by the chat, which streams; here for a run that does not.
        call = next_call(messages=messages)
        parts: list[ToolCallPart | TextPart] = (
            [TextPart(content=answer(messages=messages))]
            if call is None
            else [ToolCallPart(tool_name=call[0], args=call[1])]
        )
        input_tokens = report(messages=messages, info=info)
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=input_tokens))

    model = ProviderCounts(function=reply, model_name="a stand-in model")

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        model.sent = report(messages=messages, info=info)
        call = next_call(messages=messages)
        if call is None:
            yield answer(messages=messages)
        else:
            name, args = call
            yield {0: DeltaToolCall(name=name, json_args=args, tool_call_id=name)}

    model.stream_function = stream
    return model
