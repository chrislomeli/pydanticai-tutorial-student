"""In which order instructions from several places reach the model.

From the prompt lesson (04-04). A second agent is given instructions in
every place an agent takes them: its constructor, its capabilities, its
toolset and the run, some as text and some as functions. A stand-in model
prints each part of the instructions it is sent, in order. Kept working: a
later lesson that changes `PromptFile` or the codebase toolset updates
this script, or removes it and says why.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import PROMPT_PATH, codebase_workspace
from pydantic_tutor.codebase_tools import CodebaseDeps, codebase_toolset
from pydantic_tutor.prompt_file import PromptFile


def from_the_constructor(ctx: RunContext[CodebaseDeps]) -> str:
    """Instructions given to the constructor as a function."""
    return "From the constructor, a function."


@dataclass(kw_only=True)
class FromAFunction(AbstractCapability[CodebaseDeps]):
    """Supply `text` as instructions, from a function."""

    # What the function returns.
    text: str

    def get_instructions(self) -> Callable[[RunContext[CodebaseDeps]], str]:
        """Return a function, which the agent calls for each request."""

        def instructions(ctx: RunContext[CodebaseDeps]) -> str:
            return self.text

        return instructions


def show_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for a model: print each part of its instructions, then answer."""
    for part in info.model_request_parameters.instruction_parts or []:
        kind = "dynamic" if part.dynamic else "static"
        print(f"{kind:>7}: {part.content}")
    return ModelResponse(parts=[TextPart(content="Done.")])


# Listed so that the order you write would be wrong: a function before
# text, in the constructor and in the capabilities.
reader = Agent(
    name="reader",
    deps_type=CodebaseDeps,
    instructions=[from_the_constructor, "From the constructor, text."],
    toolsets=[codebase_toolset],
    capabilities=[
        codebase_workspace,
        FromAFunction(text="From a capability, a function."),
        PromptFile(path=PROMPT_PATH),
    ],
)
reader.run_sync(
    user_prompt="What does this codebase do?",
    model=FunctionModel(function=show_instructions),
    deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
    instructions="From the run, text.",
)
