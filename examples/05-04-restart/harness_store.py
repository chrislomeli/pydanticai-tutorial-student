"""The harness's own conversation store: what it keeps, and a stale save refused.

From 05-04-restart, "Going further": `SqliteConversationStore`, from
`pydantic_ai_harness.step_persistence.conversations`, beside the
lesson's `ConversationStore`. On a stand-in model, so it needs no key;
its file is `work/harness_conversations.db`, made new on each run. Run
it from the project root.
"""

import asyncio
from pathlib import Path

import pydantic_ai
from pydantic_ai import Agent, Conversation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SqliteConversationStore,
)

DATABASE = Path("work/harness_conversations.db")


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stand in for the model: one short answer to anything."""
    return ModelResponse(parts=[TextPart(content="Noted.")])


async def main() -> None:
    """Save a conversation, carry it on from what was saved, then save stale."""
    # What this prints is the point: no first-run banner before it.
    pydantic_ai.BANNER_ENABLED = False
    DATABASE.parent.mkdir(exist_ok=True)
    DATABASE.unlink(missing_ok=True)
    store = SqliteConversationStore(database=DATABASE)
    agent = Agent(model=FunctionModel(function=reply))

    first = await agent.run(user_prompt="My name is Ada.")
    print(f"After the first run: usage.requests={first.usage.requests}")
    # It keeps the messages, under a summary with an id and a revision of
    # its own; nothing else of the `Conversation`.
    summary = await store.save(
        summary=ConversationSummary(workspace="work/tinyjobs"),
        messages=first.all_messages(),
    )
    print(f"Saved: revision {summary.revision}")

    saved = await store.get(conversation_id=summary.id)
    conversation = Conversation(messages=saved.messages)
    print(
        f"Read back: {len(conversation.messages)} messages, "
        f"usage.requests={conversation.usage.requests}"
    )
    second = await agent.run(user_prompt="What's my name?", conversation=conversation)
    newer = await store.save(summary=saved.summary, messages=second.all_messages())
    print(f"Saved again: revision {newer.revision}")

    try:
        # As a second chat would, that read the conversation at revision 1.
        await store.save(summary=saved.summary, messages=second.all_messages())
    except ConversationConflict as error:
        print(f"Saved from revision 1 again: ConversationConflict: {error}")


asyncio.run(main=main())
