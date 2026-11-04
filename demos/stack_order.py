"""The chat's agent with its capabilities listed forwards, then reversed.

Needs no API key. It adds a `deploy.toml` with a fake token to the workspace copy;
run `demos/fresh_copy.py` first.
"""

from pydantic_ai.capabilities import AgentCapability

from pydantic_tutor.capabilities.activity_log import HookName, build_activity_hooks
from pydantic_tutor.chat.agent import build_chat_agent, chat_capabilities
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import reads_a_token_then_chains_a_command

settings = Settings()
ROOT = settings.workspace_root
LOG = settings.activity_log_path
(ROOT / "deploy.toml").write_text('runner_token = "tj-FAKE-0000-not-a-real-secret"\n')


def stack() -> list[AgentCapability[None]]:
    """The chat's capabilities, each one new."""
    return chat_capabilities(
        web_search=False,
        workspace_root=ROOT,
        activity_log=build_activity_hooks(
            path=LOG, hooks=frozenset({HookName.BEFORE_TOOL_EXECUTE})
        ),
    )


for label, capabilities in [("forwards", stack()), ("reversed", stack()[::-1])]:
    print(label)
    LOG.unlink(missing_ok=True)
    chat = build_chat_agent(
        web_search=False, workspace_root=ROOT, capabilities=capabilities
    )
    chat.run_sync(
        user_prompt="Check the deployment settings.",
        model=reads_a_token_then_chains_a_command(show_instructions=True),
    )
    print("  logged:", len(LOG.read_text().splitlines()), "tool calls")
