"""Save a conversation to a file, and load it back into a later run.

From the history lesson (02-01). Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps

HISTORY_FILE = Path("conversation.json")


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: count the turns it has seen, prompt included."""
    turn = sum(isinstance(part, UserPromptPart) for m in messages for part in m.parts)
    return ModelResponse(parts=[TextPart(content=f"this is turn {turn}")])


deps = CodebaseDeps(root=Path("sample_code/tinyjobs"))
model = FunctionModel(function=reply)

first = agent.run_sync(user_prompt="Start a conversation.", model=model, deps=deps)
print("turn 1:", first.output)

HISTORY_FILE.write_bytes(ModelMessagesTypeAdapter.dump_json(first.all_messages()))
print(f"saved {HISTORY_FILE.stat().st_size} bytes to {HISTORY_FILE}")

# A fresh process could load the file here instead; this script just reads
# back what it wrote, to show the round trip.
loaded = ModelMessagesTypeAdapter.validate_json(HISTORY_FILE.read_bytes())
print(
    "loaded:", len(loaded), "messages, same as before:", loaded == first.all_messages()
)

second = agent.run_sync(
    user_prompt="Carry on.", model=model, deps=deps, message_history=loaded
)
print("turn 2:", second.output)

HISTORY_FILE.unlink()
