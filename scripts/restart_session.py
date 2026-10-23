"""Open the chat on a stand-in model, saving its conversations in a file.

The chat is `pydantic_tutor.chat`, the one the app opens; only the model
is a stand-in, on `work/tinyjobs`, and the conversations are saved in
`work/conversations.db`, not the app's own file. Run
`scripts/fresh_copy.py` first. From 05-04-restart; a later lesson that
changes the chat keeps this script working, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.chat import chat
from pydantic_tutor.codebase_tools import CodebaseDeps
from pydantic_tutor.composition import App
from pydantic_tutor.conversations import ConversationStore


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: say which turn this is, and what you have said so far."""
    said = [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    return ModelResponse(
        parts=[TextPart(content=f"Turn {len(said)}. You've said: {' | '.join(said)}")]
    )


chat(
    app=App(
        agent=agent,
        model=FunctionModel(function=reply, model_name="a stand-in model"),
        deps=CodebaseDeps(root=Path("work/tinyjobs")),
        store=ConversationStore(path=Path("work/conversations.db")),
    )
)
