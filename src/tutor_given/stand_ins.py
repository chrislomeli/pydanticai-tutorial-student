"""Stand-in models for the lessons' keyless scripts.

Given, from the workspace lesson (03-03); `reports_what_it_is_sent` from
the toolsets lesson (04-01); `lists_reads_then_answers` and
`keeps_the_latest_exchange` from the hooks lesson (04-02);
`asks_for_every_module_at_once` and `asks_for_five_reads` from the capability
lesson (04-03); `remembers_then_recalls` and `prints_every_message` from the
memory lesson (04-05); `lists_modules_and_counts_messages` and `KeepLastTwo` from
the seam lesson (04-06). Each is a function with a readable name that returns a
model, or a capability; the lesson says what it does.
"""

import json
from collections.abc import Mapping
from dataclasses import replace

from pydantic import TypeAdapter
from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tutor_given.memory_text import MemoryListing

# What list_modules returns, checked as the stand-in models read it.
MODULE_NAMES = TypeAdapter(list[str])


def calls_tool_with_the_prompt(
    *,
    tool_name: str,
    argument: str,
    other_arguments: Mapping[str, str | int] | None = None,
) -> FunctionModel:
    """A model that calls `tool_name` with the user's prompt as `argument`.

    `other_arguments`, if given, are passed too (from the file tools lesson,
    03-04). The model then answers with the text the tool returned or the
    reason it was refused, and nothing else.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            # A refusal comes back as a RetryPromptPart, asking the model to
            # try something else; this model passes the reason on.
            if isinstance(part, ToolReturnPart | RetryPromptPart):
                return ModelResponse(parts=[TextPart(content=str(part.content))])
            if isinstance(part, UserPromptPart):
                args = {**(other_arguments or {}), argument: part.content}
                call = ToolCallPart(tool_name=tool_name, args=args)
                return ModelResponse(parts=[call])
        raise ValueError("the request has neither a prompt nor a tool result")

    return FunctionModel(function=reply)


def reports_what_it_is_sent() -> FunctionModel:
    """A model that prints the names of the tools and the instructions it is sent.

    From the toolsets lesson (04-01). It prints two lines and answers "Done."
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        print("  tools:", ", ".join(tool.name for tool in info.function_tools))
        print("  instructions:", repr(info.instructions))
        return ModelResponse(parts=[TextPart(content="Done.")])

    return FunctionModel(function=reply)


def lists_reads_then_answers() -> FunctionModel:
    """A model that makes three requests and prints what each one carries.

    From the hooks lesson (04-02). Before each request it prints how many
    messages and tools it was sent. It calls `list_modules`, then
    `read_docstring` on `tinyjobs.jobs`, then answers "Done.".
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        print(
            f"  the model receives: messages={len(messages)} "
            f"tools={len(info.function_tools)}"
        )
        returned = [
            part.tool_name
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returned:
            call = ToolCallPart(tool_name="list_modules", args={})
            return ModelResponse(parts=[call])
        if returned == ["list_modules"]:
            call = ToolCallPart(
                tool_name="read_docstring", args={"module": "tinyjobs.jobs"}
            )
            return ModelResponse(parts=[call])
        return ModelResponse(parts=[TextPart(content="Done.")])

    return FunctionModel(function=reply)


def keeps_the_latest_exchange(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """A `before_model_request` hook that sends the question and the latest exchange only.

    From the hooks lesson (04-02): a stand-in for a history trimmer. Up to
    three messages it passes the request on unchanged; after that it sends
    the first message and the last two (a tool's call and its return).
    """
    messages = request_context.messages
    if len(messages) <= 3:
        return request_context
    # A new request with a shorter list: the run's history keeps every message.
    return replace(request_context, messages=[messages[0], *messages[-2:]])


def asks_for_every_module_at_once() -> FunctionModel:
    """A model that lists the modules, then asks to read every unread one in one response.

    From the capability lesson (04-03). Each time it prints what it asks
    for, and it answers "I read N modules." once every module has a
    docstring returned.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        listed: list[str] = []
        read = 0
        for message in messages:
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                if part.tool_name == "list_modules":
                    listed = MODULE_NAMES.validate_python(part.content)
                elif part.tool_name == "read_docstring":
                    read += 1
        if not listed:
            print("the model asks for the module list")
            call = ToolCallPart(tool_name="list_modules", args={})
            return ModelResponse(parts=[call])
        # Each docstring returned is one module read, in the order asked for.
        unread = listed[read:]
        if not unread:
            print("the model answers")
            return ModelResponse(parts=[TextPart(content=f"I read {read} modules.")])
        print(f"the model asks to read {len(unread)} modules: {', '.join(unread)}")
        calls = [
            ToolCallPart(tool_name="read_docstring", args={"module": module})
            for module in unread
        ]
        return ModelResponse(parts=calls)

    return FunctionModel(function=reply)


def asks_for_five_reads() -> FunctionModel:
    """A model that asks for five docstrings in one response, then answers "Done.".

    From the capability lesson (04-03). It prints one line each time, and
    asks for the five modules of `tinyjobs` whatever the workspace holds.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            print("  the model answers")
            return ModelResponse(parts=[TextPart(content="Done.")])
        print("  the model asks for 5 tool calls")
        modules = (
            "tinyjobs",
            "tinyjobs.cli",
            "tinyjobs.jobs",
            "tinyjobs.log",
            "tinyjobs.storage",
        )
        calls = [
            ToolCallPart(tool_name="read_docstring", args={"module": module})
            for module in modules
        ]
        return ModelResponse(parts=calls)

    return FunctionModel(function=reply)


def remembers_then_recalls() -> FunctionModel:
    """A model that stores a note when told to remember, and looks one up when asked.

    From the memory lesson (04-05). A prompt starting "Remember" makes it
    call `write_memory` with a note about logging; any other prompt makes
    it call `list_memory`, `read_memory` on the first file listed, and
    answer with what it read. It prints what it is sent and what it asks for.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Each request the model is sent: its parts' kinds. With injection
        # off, no part carries the notebook.
        print("  request", [part.part_kind for part in messages[-1].parts])
        for part in messages[-1].parts:
            if isinstance(part, UserPromptPart):
                if str(part.content).startswith("Remember"):
                    note = "- Logging goes to stderr, so stdout stays pipeable.\n"
                    return call(
                        tool="write_memory",
                        args={"content": note, "file": "logging.md"},
                    )
                return call(tool="list_memory", args={})
            if isinstance(part, ToolReturnPart):
                print("  result", part.model_response_str())
                if part.tool_name == "list_memory":
                    listing = MemoryListing.model_validate_json(
                        part.model_response_str()
                    )
                    return call(tool="read_memory", args={"file": listing.files[0]})
                if part.tool_name == "read_memory":
                    note = part.model_response_str().strip()
                    return ModelResponse(
                        parts=[TextPart(content=f"We decided: {note}")]
                    )
                return ModelResponse(parts=[TextPart(content="Noted.")])
        raise ValueError("a request this stand-in was not written for")

    def call(*, tool: str, args: dict[str, str]) -> ModelResponse:
        print("  call  ", tool, json.dumps(obj=args))
        return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])

    return FunctionModel(function=reply)


def prints_every_message() -> FunctionModel:
    """A model that prints each message of the request it is sent, then answers "ok".

    From the memory lesson (04-05). Each message prints on one line: its
    kind, then the text of each part. A part of another kind shows as its
    kind, and a part made of several text items shows them joined.
    """

    def text_of(*, part: ModelRequestPart | ModelResponsePart) -> str:
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
            text = " ".join(
                item.content for item in part.content if isinstance(item, TextContent)
            )
        elif isinstance(part, UserPromptPart | TextPart):
            text = str(part.content)
        else:
            text = part.part_kind
        return " ".join(text.split())

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            print(f"  {message.kind}:", [text_of(part=part) for part in message.parts])
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(function=reply)


class KeepLastTwo(AbstractCapability[None]):
    """A capability that sends the model only the last two messages of each request.

    From the seam lesson (04-06): a stand-in for one that shortens a long
    request, as a later lesson's compaction will.
    """

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Return the request with its earlier messages left out."""
        # Two, so a tool call is sent with its return. A copy: the run's
        # history keeps every message.
        return replace(request_context, messages=request_context.messages[-2:])


def lists_modules_and_counts_messages() -> FunctionModel:
    """A model that says what it is sent, asks for `list_modules` once, then answers.

    From the seam lesson (04-06). On each request it prints how many messages
    it is sent; on the first, how each piece of its instructions starts.
    """

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        print(f"  model: sent {len(messages)} message(s)")
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content="Done.")])
        for part in info.model_request_parameters.instruction_parts or []:
            print("    instructions:", " ".join(part.content.split()[:4]), "...")
        return ModelResponse(parts=[ToolCallPart(tool_name="list_modules", args={})])

    return FunctionModel(function=reply)
