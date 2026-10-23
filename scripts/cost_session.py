"""Open the chat on a stand-in model that says what each request carried, and counts it as a provider would.

The chat is `pydantic_tutor.chat`, the one the app opens, on
`work/tinyjobs`; nothing is saved. Asked anything, the stand-in model
lists the modules, reads one docstring, then answers: three requests a
turn. For each request it prints the tokens of its three parts: the
instructions, the tool definitions and the messages, at about four
characters a token. It reports their sum as the request's input tokens,
as a provider reports what it counted. Run `scripts/fresh_copy.py`
first. From 06-01-cost; a later lesson that changes the chat keeps this
script working, or removes it and says why.
"""

import json
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from pydantic_tutor.agent import agent
from pydantic_tutor.chat import chat
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.composition import App

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


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: say what it was sent, then list, read and answer.

    It reads the turn so far, from its last prompt on, and asks for the
    first of its two tools that has not returned in it yet.
    """
    instructions = len(info.instructions or "") // CHARS_PER_TOKEN
    tools = (
        sum(len(definition_text(tool=tool)) for tool in info.function_tools)
        // CHARS_PER_TOKEN
    )
    history = (
        sum(len(message_text(message=message)) for message in messages)
        // CHARS_PER_TOKEN
    )
    number = 1 + sum(isinstance(message, ModelResponse) for message in messages)
    print(
        f"  request {number}: instructions {instructions:,} · "
        f"tool definitions {tools:,} ({len(info.function_tools)}) · "
        f"messages {history:,} ({len(messages)})"
    )
    # The request that opens this turn: a prompt and no tool return. (A
    # tool a load reveals is announced to this stand-in as a prompt too.)
    start = max(
        index
        for index, message in enumerate(messages)
        if any(isinstance(part, UserPromptPart) for part in message.parts)
        and not any(isinstance(part, ToolReturnPart) for part in message.parts)
    )
    prompt = "".join(
        str(part.content)
        for part in messages[start].parts
        if isinstance(part, UserPromptPart)
    )
    returned = {
        part.tool_name: part.model_response_str()
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    sent_tools = {tool.name for tool in info.function_tools}
    parts: list[ToolCallPart | TextPart]
    if prompt.startswith("Remember that "):
        # Asked to remember: load the notebook if its tools weren't sent,
        # write the note, then answer.
        if "write_memory" not in sent_tools:
            parts = [ToolCallPart(tool_name="load_capability", args={"id": "memory"})]
        elif "write_memory" not in returned:
            note = f"- {prompt.removeprefix('Remember that ')}\n"
            parts = [ToolCallPart(tool_name="write_memory", args={"content": note})]
        else:
            parts = [TextPart(content="Noted.")]
    elif "list_modules" not in returned:
        parts = [ToolCallPart(tool_name="list_modules", args={})]
    elif "read_docstring" not in returned:
        parts = [
            ToolCallPart(tool_name="read_docstring", args={"module": "tinyjobs.jobs"})
        ]
    else:
        summary = returned["read_docstring"].splitlines()[0]
        parts = [TextPart(content=f"tinyjobs.jobs: {summary}")]
    output = len(message_text(message=ModelResponse(parts=parts))) // CHARS_PER_TOKEN
    # What a provider does: report the tokens it was sent, all of them,
    # and the tokens of its answer.
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(
            input_tokens=instructions + tools + history, output_tokens=output
        ),
    )


chat(
    app=App(
        agent=agent,
        model=FunctionModel(function=reply, model_name="a stand-in model"),
        deps=CodebaseDeps(root=Path("work/tinyjobs")),
    )
)
