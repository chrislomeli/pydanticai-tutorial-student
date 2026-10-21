"""What memory injection would send: the notebook beside every question.

From the memory lesson (04-05). A second agent, with the library's
`Memory` and injection on, reads the notebook `scripts/notebook.py` left
in `work/tinyjobs`. A conversation of two turns, on a stand-in model that
prints every message of each request it is sent. The project's agent keeps
injection off; this shows what that saves. Kept working: a later lesson
that changes the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore

from pydantic_tutor.agent import repo_workspace
from pydantic_tutor.repo_tools import RepoDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: print each message of the request, then answer."""
    for message in messages:
        print(f"  {message.kind}:", [text_of(part=part) for part in message.parts])
    return ModelResponse(parts=[TextPart(content="ok")])


def text_of(*, part: ModelRequestPart | ModelResponsePart) -> str:
    """A part's text, on one line: the question, the answer, or the memory block."""
    if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
        # The memory block is a user part holding text items, not a string.
        text = " ".join(
            item.content for item in part.content if isinstance(item, TextContent)
        )
    elif isinstance(part, UserPromptPart | TextPart):
        text = str(part.content)
    else:
        text = part.part_kind
    return " ".join(text.split())


# The same notebook as the project's agent: the same folder in the same
# workspace. Only `inject_memory` differs.
injecting = Agent(
    deps_type=RepoDeps,
    capabilities=[
        repo_workspace,
        Memory(store=FileStore(directory=".agent-memory"), inject_memory=True),
    ],
)

model = FunctionModel(function=reply)
deps = RepoDeps(root=Path("work/tinyjobs"))
history: list[ModelMessage] = []
for prompt in ["What does tinyjobs do?", "And how is it tested?"]:
    print(f"turn: {prompt}")
    result = injecting.run_sync(
        user_prompt=prompt, model=model, deps=deps, message_history=history
    )
    history = result.all_messages()
