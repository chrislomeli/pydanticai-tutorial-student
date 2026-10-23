"""Two sessions with the agent: one stores a note, the next finds it.

From the memory lesson (04-05). Runs the agent twice on `work/tinyjobs`,
with no history between the runs, on a stand-in model that uses the memory
tools as a model might. Prints each request the model is sent, and each
tool call and its result. Run `scripts/fresh_copy.py` first: the notebook
is kept in the copy, and grows with every run. Kept working: a later
lesson that changes the agent updates this script, or removes it and says
why.
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

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.listable_memory import MemoryListing


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: store a note when told to, and look one up when asked."""
    request = messages[-1]  # the latest message: the request to answer
    # Each request the model is sent: its parts' kinds. With injection
    # off, no part carries the notebook.
    print("  request", [part.part_kind for part in request.parts])
    for part in request.parts:
        if isinstance(part, UserPromptPart):
            if str(part.content).startswith("Remember"):
                note = "- Logging goes to stderr, so stdout stays pipeable.\n"
                return call(
                    tool="write_memory", args={"content": note, "file": "logging.md"}
                )
            return call(tool="list_memory", args={})
        if isinstance(part, ToolReturnPart):
            print("  result", part.model_response_str())
            if part.tool_name == "list_memory":
                # Read the first file the listing names, as a model would.
                listing = MemoryListing.model_validate_json(part.model_response_str())
                return call(tool="read_memory", args={"file": listing.files[0]})
            if part.tool_name == "read_memory":
                return answer(text=f"We decided: {part.model_response_str().strip()}")
            return answer(text="Noted.")
    raise ValueError("a request this stand-in was not written for")


def call(*, tool: str, args: dict[str, str]) -> ModelResponse:
    """A response asking for one tool call, printed as the model would send it."""
    print("  call  ", tool, json.dumps(obj=args))
    return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args)])


def answer(*, text: str) -> ModelResponse:
    """A response that ends the run with `text`."""
    return ModelResponse(parts=[TextPart(content=text)])


model = FunctionModel(function=reply)
deps = CodebaseDeps(root=Path("work/tinyjobs"))
for prompt in ["Remember that we log to stderr.", "What did we decide about logging?"]:
    # A new run with no message_history: a new session, as far as the
    # model can tell. Only the notebook's files carry over.
    print(f"session: {prompt}")
    result = agent.run_sync(user_prompt=prompt, model=model, deps=deps)
    print("  output", result.output)
