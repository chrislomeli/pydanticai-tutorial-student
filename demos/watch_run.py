"""Run the chat's agent on a stand-in model, and print its activity log.

Needs no API key; run `demos/fresh_copy.py` first. It starts the log again,
so the file holds this run only.
"""

from pydantic_ai.models.test import TestModel

from pydantic_tutor.capabilities.activity_log import build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.paths import project_root

# Beside the workspace copy and not in it, so the file tools can't reach it.
# `work/` is already ignored by git.
LOG_PATH = project_root() / "work" / "activity.jsonl"
LOG_PATH.unlink(missing_ok=True)

chat = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
chat.run_sync(
    user_prompt="What does this project do?",
    model=TestModel(call_tools=["list_modules", "write_file"]),
    # Added for this run only: the chat itself doesn't change.
    capabilities=[build_activity_hooks(path=LOG_PATH)],
)
print(LOG_PATH.read_text())
