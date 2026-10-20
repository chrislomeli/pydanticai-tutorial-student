"""Ask for two modules the tool cannot read: one misspelt, one broken.

From the limits lesson (01-08). It makes a small repository of its own, in
a temporary folder, with a module that does not parse. Kept working: a
later lesson that changes the tools updates this script, or removes it
and says why.
"""

import tempfile
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.repo_tools import RepoDeps

# The modules the model asks for, in order: a misspelling, then the real one.
ASKS = ("brokn", "broken")


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: ask for each of ASKS, print what comes back, then answer."""
    for part in messages[-1].parts:
        if isinstance(part, RetryPromptPart):
            print("asked to retry:", part.model_response())
        if isinstance(part, ToolReturnPart):
            print(f"tool result ({part.outcome}):", part.model_response_str())
    asked = sum(isinstance(message, ModelResponse) for message in messages)
    if asked < len(ASKS):
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_docstring", args={"module": ASKS[asked]})
            ]
        )
    return ModelResponse(parts=[TextPart(content="The module broken does not parse.")])


with tempfile.TemporaryDirectory() as folder:
    package = Path(folder) / "src" / "broken"
    package.mkdir(parents=True)
    # Half-written code: a definition left open.
    (package / "__init__.py").write_text(data="def half_written(:\n")
    result = agent.run_sync(
        user_prompt="What does the broken module do?",
        model=FunctionModel(function=reply),
        deps=RepoDeps(root=Path(folder)),
    )
print("output:", result.output)
