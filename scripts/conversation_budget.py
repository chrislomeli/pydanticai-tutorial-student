"""Carry a dialogue forward with `conversation=`, and watch a budget hold.

From the Conversation lesson (02-02). Kept working: a later lesson that
changes the agent updates this script, or removes it and says why.
"""

from pathlib import Path

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: count the turns it has seen, prompt included."""
    turn = sum(isinstance(part, UserPromptPart) for m in messages for part in m.parts)
    return ModelResponse(parts=[TextPart(content=f"this is turn {turn}")])


deps = CodebaseDeps(root=Path("sample_code/tinyjobs"))
model = FunctionModel(function=reply)
limits = UsageLimits(request_limit=2)

print("conversation=:")
conversation = None
outcomes_with_conversation: list[str] = []
for turn in range(1, 5):
    try:
        result = agent.run_sync(
            user_prompt=f"turn {turn}",
            model=model,
            deps=deps,
            conversation=conversation,
            usage_limits=limits,
        )
        conversation = result.conversation
        outcomes_with_conversation.append("ok")
        print(
            f"  turn {turn}: ok"
            f"  conversation_id={conversation.conversation_id}"
            f"  usage.requests={conversation.usage.requests}"
        )
    except UsageLimitExceeded:
        outcomes_with_conversation.append("LIMIT")
        print(f"  turn {turn}: LIMIT")
print(f"  turns 1-4: {outcomes_with_conversation}")

print("\nmessage_history=:")
history: list[ModelMessage] = []
outcomes_with_history: list[str] = []
for turn in range(1, 5):
    try:
        result = agent.run_sync(
            user_prompt=f"turn {turn}",
            model=model,
            deps=deps,
            message_history=history,
            usage_limits=limits,
        )
        history = result.all_messages()
        outcomes_with_history.append("ok")
        print(f"  turn {turn}: ok  usage.requests={result.usage.requests}")
    except UsageLimitExceeded:
        outcomes_with_history.append("LIMIT")
        print(f"  turn {turn}: LIMIT")
print(f"  turns 1-4: {outcomes_with_history}")
