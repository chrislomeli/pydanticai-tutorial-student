"""The chat's agent with its capabilities listed forwards, then reversed.

From the seam lesson (04-06). Needs no API key; run `demos/fresh_copy.py`
first. Kept working: a later lesson that changes the chat's stack updates
this script, or removes it and says why.
"""

import logging
import sys

from pydantic_ai.capabilities import AgentCapability

from pydantic_tutor.chat.agent import build_chat_agent, chat_capabilities
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import (
    KeepLastTwo,
    lists_modules_and_counts_messages,
)

# The request log on, printing on stdout in order with this script's lines.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.capabilities.request_log").setLevel(
    level=logging.INFO
)

ROOT = Settings().workspace_root


def stack() -> list[AgentCapability[None]]:
    """The chat's capabilities, each one new, and the stand-in last."""
    return [*chat_capabilities(web_search=False, workspace_root=ROOT), KeepLastTwo()]


for label, capabilities in [("forwards", stack()), ("reversed", stack()[::-1])]:
    print(label)
    chat = build_chat_agent(
        web_search=False, workspace_root=ROOT, capabilities=capabilities
    )
    chat.run_sync(
        user_prompt="What does this project do?",
        model=lists_modules_and_counts_messages(),
    )
