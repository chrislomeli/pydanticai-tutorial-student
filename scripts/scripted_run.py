"""Run the agent on a model you write: a function that answers from the request.

From the agent lesson (01-02). Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.repo_tools import RepoDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: answer from what the model is sent.

    FunctionModel calls this once for each request the agent makes, with
    every message of the run so far, and `info`, which carries what is sent
    beside the messages, such as the agent's instructions. What it returns
    is what the model said.
    """
    request = messages[-1]  # the latest message: the request to answer
    # A request is made of parts; the user's prompt is one of them.
    for part in request.parts:
        if isinstance(part, UserPromptPart):
            text = f"You asked {part.content!r}, and my instructions say: {info.instructions}"
            # One text part: the model's whole answer, which becomes result.output.
            return ModelResponse(parts=[TextPart(content=text)])
    # Every request in this lesson carries the prompt. Once the agent has
    # tools, a request can carry only a tool's result, and this function
    # will have to answer that too.
    raise ValueError("the request has no user prompt")


# The model is given for this run only; the agent itself has none (step 1).
result = agent.run_sync(
    user_prompt="What does this repo do?",
    model=FunctionModel(function=reply),
    deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
)
print(result.output)
