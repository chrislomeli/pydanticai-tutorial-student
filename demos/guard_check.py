"""Run the chat's agent on a model that reads a token and chains a command.

Needs no API key. It resets the log, and adds a `deploy.toml` with a fake token
to the workspace copy; run `demos/fresh_copy.py` first.
"""

from pydantic_tutor.capabilities.activity_log import HookName, build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import reads_a_token_then_chains_a_command

settings = Settings()
(settings.workspace_root / "deploy.toml").write_text(
    'runner_token = "tj-FAKE-0000-not-a-real-secret"\n'
)
log_path = settings.activity_log_path
log_path.unlink(missing_ok=True)

chat = build_chat_agent(
    web_search=False,
    workspace_root=settings.workspace_root,
    activity_log=build_activity_hooks(
        path=log_path,
        hooks=frozenset({HookName.BEFORE_TOOL_EXECUTE, HookName.AFTER_TOOL_EXECUTE}),
    ),
)
chat.run_sync(
    user_prompt="Check the deployment settings.",
    model=reads_a_token_then_chains_a_command(),
)
print("the log:")
print(log_path.read_text(), end="")
