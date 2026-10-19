"""Print what the model is told about the agent's tools.

From the function tools lesson (01-03). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

import json
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.repo_tools import RepoDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: print the tool definitions it is sent, then answer."""
    # info.function_tools holds one definition per tool: all the model
    # knows of it.
    for tool in info.function_tools:
        print(tool.name)
        print("  description:", tool.description)
        print("  parameters:", json.dumps(obj=tool.parameters_json_schema, indent=2))
    return ModelResponse(parts=[TextPart(content="Done.")])


agent.run_sync(
    user_prompt="What does the jobs module do?",
    model=FunctionModel(function=reply),
    deps=RepoDeps(root=Path("sample_repos/tinyjobs")),
)
