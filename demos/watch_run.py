"""Run the chat's agent on a stand-in model, and print its activity log.

Needs no API key; run `demos/fresh_copy.py` first. It starts the log again,
so the file holds this run only.
"""

from pydantic_ai.models.test import TestModel

from pydantic_tutor.capabilities.activity_log import build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings

settings = Settings()
settings.activity_log_path.unlink(missing_ok=True)

chat = build_chat_agent(
    web_search=False,
    workspace_root=settings.workspace_root,
    activity_log=build_activity_hooks(
        path=settings.activity_log_path, hooks=settings.activity_log
    ),
)
chat.run_sync(
    user_prompt="What does this project do?",
    model=TestModel(call_tools=["list_modules", "write_file"]),
)
print(
    settings.activity_log_path.read_text()
    if settings.activity_log_path.exists()
    else "(nothing recorded)"
)
