"""In which order instructions from several places reach the model.

From the prompt lesson (04-04). A second agent is given instructions in
every place an agent takes them: its constructor, a capability, a toolset
and the run, some as text and some as a function. A stand-in model prints
each part of the instructions it is sent, in order. Needs no API key.
Kept working: a later lesson that changes `PromptFile` or the workspace
toolset updates it, or says why not.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.capabilities.prompt_file import PromptFile
from pydantic_tutor.chat.agent import PROMPT_PATH
from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_tools import build_workspace_toolset


def from_the_constructor(ctx: RunContext[None]) -> str:
    """Instructions given to the constructor as a function."""
    return "From the constructor, a function."


def show_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: print each part of its instructions, then answer."""
    for part in info.model_request_parameters.instruction_parts or []:
        kind = "dynamic" if part.dynamic else "static"
        print(f"{kind:>7}: {part.content}")
    return ModelResponse(parts=[TextPart(content="Done.")])


# The function is listed before the text, so that the order you write would
# be wrong.
reader: Agent[None, str] = Agent(
    name="reader",
    instructions=[from_the_constructor, "From the constructor, text."],
    toolsets=[build_workspace_toolset(root=Settings().workspace_root)],
    capabilities=[PromptFile(path=PROMPT_PATH)],
)
reader.run_sync(
    user_prompt="Hi.",
    model=FunctionModel(function=show_instructions),
    instructions="From the run, text.",
)
