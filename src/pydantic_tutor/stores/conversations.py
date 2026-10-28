"""The chat's conversations: what a store must do, and what it returns.

Ours: Pydantic AI turns messages into JSON (`ModelMessagesTypeAdapter`) and
leaves where you keep them to you. The key is the `conversation_id` they
carry. `ConversationStore` is the interface; `SyncSqliteConversationStore`
(`sqlite_conversations.py`) keeps them in a SQLite file.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage


class ConversationConflictError(Exception):
    """Another save of this conversation came first; saving would overwrite its turns."""


@dataclass(frozen=True, kw_only=True)
class Saved:
    """A conversation read from the store, and the revision it was read at."""

    messages: list[ModelMessage]
    revision: int


class ConversationStore(ABC):
    """Where the chat's conversations are kept."""

    @abstractmethod
    def save(self, *, messages: list[ModelMessage], revision: int) -> int:
        """Save `messages` if their row is still at `revision` (0: never saved).

        Returns the new revision. Raises ConversationConflictError, saving
        nothing, if another save of the conversation came first.
        """

    @abstractmethod
    def latest(self) -> Saved | None:
        """The conversation saved most recently, or None if none has been."""
