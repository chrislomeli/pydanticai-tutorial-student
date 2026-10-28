"""Run the chat's agent on a stand-in model, with its request log on.

From the hooks lesson (04-02). Needs no API key; run `demos/fresh_copy.py`
first. Kept working: a later lesson that changes the agent updates it, or
removes it and says why.
"""

import logging
import sys

from pydantic_ai.models.test import TestModel

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings

# The program that runs the agent decides which logs are shown, and where:
# here, our request log, on stdout, in order with this script's own lines.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.capabilities.request_log").setLevel(
    level=logging.INFO
)

chat = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
# TestModel calls list_modules, then answers: a run of two requests.
result = chat.run_sync(
    user_prompt="What does this project do?",
    model=TestModel(call_tools=["list_modules"]),
)
print("output:", result.output)
print("usage:", result.usage)
