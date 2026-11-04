"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes one on
each run.

Built from these pieces:
- Workspace: the folder the tools work in.
- File tools: read and write files there.
- Shell tools: run a few commands there.
- Toolset: the workspace's own tools.
- Web search: switched on by a setting.
- Activity log: a record of the run, if one is given.
- Guards: our rules for tool calls.
- Token redaction: secrets removed before the model sees them.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    LocalWorkspace,
    Toolset,
    WebSearch,
)
from pydantic_ai_harness import FileSystem, Shell

from pydantic_tutor.capabilities.guards import guards
from pydantic_tutor.capabilities.redact_tokens import RedactTokens
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The commands the shell may start. A precaution, not a boundary: it checks
# only a command's first word, so `ls && cat .env` still runs.
SHELL_COMMANDS = ("pytest", "pwd", "ls", "cat")

# The shape of this project's tokens.
TOKEN_PATTERN = r"tj-[A-Za-z0-9-]+"


def build_chat_agent(
    *, web_search: bool, workspace_root: Path, activity_log: Hooks[None] | None = None
) -> Agent[None, str]:
    """Build the chat's agent."""
    capabilities: list[AbstractCapability[None]] = [
        # Workspace: a place for tools to work in. It adds no tool: they use
        # `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
        # File tools: read and write files in that workspace, minus the secret
        # patterns.
        FileSystem(denied_patterns=SECRET_PATTERNS),
        # Shell tools: run the allowed commands in that workspace.
        Shell(allowed_commands=SHELL_COMMANDS),
        # Toolset: the workspace's own tools, built for this folder.
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
        # Our rules for tool calls: see the module's limits.
        guards,
        RedactTokens(pattern=TOKEN_PATTERN),
    ]
    if web_search:
        # Web search: the model may search when it needs to. Each search is
        # billed by the provider, on top of the tokens.
        capabilities.append(WebSearch())
    if activity_log is not None:
        # First: its `before_` hooks see every attempt, and its `after_` hooks
        # run last, so they see the result the model gets.
        capabilities.insert(0, activity_log)
    return Agent[None, str](
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# Earlier demos import this name; the console chat builds its agent in build_app.
chat_agent = build_chat_agent(
    web_search=True, workspace_root=project_root() / "work" / "tinyjobs"
)
