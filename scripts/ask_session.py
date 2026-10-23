"""Open the chat on a stand-in model that asks you a question each turn.

The chat is `pydantic_tutor.chat`, the one the app opens; only the model
is a stand-in, on `work/tinyjobs`. Run `scripts/fresh_copy.py` first.
From 05-03-ask-user; a later lesson that changes the chat keeps this
script working, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.ask_user import TOOL_NAME, Question, QuestionOption

from pydantic_tutor.agent import agent
from pydantic_tutor.chat import chat
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.composition import App

# The question the stand-in model asks, in the harness's own model of
# one: an edit that breaks the tool's limits fails here, not mid-chat.
QUESTION = Question(
    header="Flag",
    question="Which flag should the CLI gain?",
    options=(
        QuestionOption(label="--verbose", description="print each job as it runs"),
        QuestionOption(label="--quiet", description="print only errors"),
    ),
)


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model, whatever was typed.

    Each turn asks the question, then says what the tool returned: the
    answer, as the model is sent it.
    """
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        # The tool's arguments, as a model sends them: plain JSON.
        args = {"questions": [QUESTION.model_dump(mode="json")]}
        call = ToolCallPart(tool_name=TOOL_NAME, args=args)
        return ModelResponse(parts=[call])
    return ModelResponse(parts=[TextPart(content=f"I was told: {returns[0].content}")])


chat(
    app=App(
        agent=agent,
        model=FunctionModel(function=reply, model_name="a stand-in model"),
        deps=CodebaseDeps(root=Path("work/tinyjobs")),
    )
)
