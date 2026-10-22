"""The chat's conversations, saved in a SQLite file, so a new process can carry one on.

Ours, not the library's: Pydantic AI turns a `Conversation` into JSON and
back (`ConversationTypeAdapter`), and leaves where you keep it to you.
Here, one row per conversation, keyed by its `conversation_id`. A save
replaces the row only if no other save of it came after it was read: two
chats on one conversation never silently overwrite each other's turns.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Conversation, ConversationTypeAdapter

# `revision` counts a conversation's saves; `saved_order` numbers every
# save in the file, so the highest is the one saved last.
SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_conversations (
    conversation_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    saved_order INTEGER NOT NULL,
    conversation TEXT NOT NULL
)
"""


class ConversationConflictError(Exception):
    """Another save of this conversation came first; saving would overwrite its turns."""


@dataclass(frozen=True, kw_only=True)
class Saved:
    """A conversation read from the store, and the revision it was read at."""

    conversation: Conversation
    revision: int


@dataclass(frozen=True, kw_only=True)
class ConversationStore:
    """Every conversation saved in one SQLite file, the whole `Conversation` each.

    The file and its table are made on first use, not when the store is built.
    """

    path: Path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection to the file, its table made if missing; committed, then closed."""
        # `closing` closes the connection; its own `with` commits, or rolls
        # back if the block raised.
        with (
            closing(thing=sqlite3.connect(database=self.path)) as connection,
            connection,
        ):
            connection.execute(SCHEMA)
            yield connection

    def save(self, *, conversation: Conversation, revision: int) -> int:
        """Save `conversation` whole, if its row is still at `revision`; return the new one.

        `revision` is the one `latest()` read it at, or 0 if it was never
        saved. Raises ConversationConflictError, saving nothing, if another
        save of it came first.
        """
        data = ConversationTypeAdapter.dump_json(conversation).decode()
        with self._connect() as connection:
            if revision == 0:
                cursor = connection.execute(
                    "INSERT INTO chat_conversations VALUES (?, 1, (SELECT "
                    "coalesce(max(saved_order), 0) + 1 FROM chat_conversations), ?) "
                    "ON CONFLICT (conversation_id) DO NOTHING",
                    (conversation.conversation_id, data),
                )
            else:
                cursor = connection.execute(
                    "UPDATE chat_conversations SET revision = revision + 1, "
                    "saved_order = (SELECT "
                    "coalesce(max(saved_order), 0) + 1 FROM chat_conversations), "
                    "conversation = ? "
                    "WHERE conversation_id = ? AND revision = ?",
                    (data, conversation.conversation_id, revision),
                )
        if cursor.rowcount != 1:
            raise ConversationConflictError(
                f"conversation {conversation.conversation_id} was saved again "
                f"after revision {revision} was read"
            )
        return revision + 1

    def latest(self) -> Saved | None:
        """The conversation saved most recently, or None if none has been."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation, revision FROM chat_conversations "
                "ORDER BY saved_order DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data, revision = row
        return Saved(
            conversation=ConversationTypeAdapter.validate_json(data), revision=revision
        )
