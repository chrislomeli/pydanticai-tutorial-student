"""The chat's conversations in a SQLite file: one row each, its messages as JSON.

Ours: the `ConversationStore` (`conversations.py`) that keeps them in one
SQLite file. Another store, in memory or in a server database, is another
subclass; nothing else in the chat changes.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pydantic_tutor.stores.conversations import (
    ConversationConflictError,
    ConversationStore,
    Saved,
)

# `revision` counts a conversation's saves; `saved_at` says which was last.
SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_conversations (
    conversation_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    saved_at TEXT NOT NULL,
    messages TEXT NOT NULL
)
"""

# Insert; or, if the row exists, update it only while it is at the revision read.
SAVE = """
INSERT INTO chat_conversations (conversation_id, revision, saved_at, messages)
VALUES (:id, 1, :now, :messages)
ON CONFLICT (conversation_id) DO UPDATE
SET revision = revision + 1, saved_at = :now, messages = :messages
WHERE revision = :revision
"""


@dataclass(frozen=True, kw_only=True)
class SyncSqliteConversationStore(ConversationStore):
    """Every conversation saved in one SQLite file. The file is made on first use."""

    path: Path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection with the table made; committed, or rolled back, then closed."""
        with (
            closing(thing=sqlite3.connect(database=self.path)) as connection,
            connection,
        ):
            connection.execute(SCHEMA)
            yield connection

    def save(self, *, messages: list[ModelMessage], revision: int) -> int:
        """Save `messages` if their row is still at `revision` (0: never saved).

        Returns the new revision. Raises ConversationConflictError, saving
        nothing, if another save of the conversation came first.
        """
        conversation_id = messages[-1].conversation_id if messages else None
        if conversation_id is None:
            raise ValueError("no messages with a conversation_id to save")
        with self._connect() as connection:
            cursor = connection.execute(
                SAVE,
                {
                    "id": conversation_id,
                    "now": datetime.now(tz=UTC).isoformat(),
                    "messages": ModelMessagesTypeAdapter.dump_json(messages).decode(),
                    "revision": revision,
                },
            )
        if cursor.rowcount != 1:
            raise ConversationConflictError(
                f"conversation {conversation_id} was saved again "
                f"after revision {revision} was read"
            )
        return revision + 1

    def latest(self) -> Saved | None:
        """The conversation saved most recently, or None if none has been."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT messages, revision FROM chat_conversations "
                "ORDER BY saved_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data, revision = row
        return Saved(
            messages=ModelMessagesTypeAdapter.validate_json(data), revision=revision
        )
