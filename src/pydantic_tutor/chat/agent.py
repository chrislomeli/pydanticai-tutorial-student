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

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AgentCapability,
    Hooks,
    LocalWorkspace,
    Toolset,
    WebSearch,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.memory import FileStore

from pydantic_tutor.capabilities.guards import guards
from pydantic_tutor.capabilities.listable_memory import ListableMemory
from pydantic_tutor.capabilities.prompt_file import PromptFile
from pydantic_tutor.capabilities.redact_tokens import RedactTokens
from pydantic_tutor.tools.workspace_toolset import build_workspace_toolset
from tutor_given.paths import project_root

# File names the file tools may not touch. A precaution, not a complete
# data-security policy.
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The notebook's folder, in the workspace. The file tools may not touch it: the
# memory tools are its only way in.
NOTEBOOK = ".agent-memory"

# The commands the shell may start. A precaution, not a boundary: it checks
# only a command's first word, so `ls && cat .env` still runs.
SHELL_COMMANDS = ("pytest", "pwd", "ls", "cat")

# The shape of this project's tokens.
TOKEN_PATTERN = r"tj-[A-Za-z0-9-]+"

# The chat's prompt, in the package's `prompts` folder, so it is found whichever
# folder the program runs from.
PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "chat.md"


def chat_capabilities(
    *, web_search: bool, workspace_root: Path, activity_log: Hooks[None] | None = None
) -> list[AgentCapability[None]]:
    """Everything the chat does, as a list of capabilities, each one new."""
    capabilities: list[AgentCapability[None]] = [
        # The chat's instructions, read from PROMPT_PATH when the agent is
        # built. Text, so it is sent ahead of any computed for a request.
        PromptFile(path=PROMPT_PATH),
        # Workspace: a place for tools to work in. It adds no tool: they use
        # `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
        # File tools: read and write files in that workspace, minus the secret
        # patterns.
        # The notebook is denied too, for reads and writes alike.
        FileSystem(denied_patterns=(*SECRET_PATTERNS, f"**/{NOTEBOOK}/**")),
        # Shell tools: run the allowed commands in that workspace.
        Shell(allowed_commands=SHELL_COMMANDS),
        # Toolset: the workspace's own tools, built for this folder.
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
        # Injection off: the model lists and reads the notebook with tools.
        ListableMemory(store=FileStore(directory=NOTEBOOK), inject_memory=False),
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
    return capabilities


def build_chat_agent(
    *,
    web_search: bool,
    workspace_root: Path,
    activity_log: Hooks[None] | None = None,
    capabilities: Sequence[AgentCapability[None]] | None = None,
) -> Agent[None, str]:
    """Build the chat's agent: what it is here, what it does in `capabilities`.

    The constructor sets what no capability can. `capabilities` defaults to
    `chat_capabilities(...)`; pass another list to rearrange or extend the
    stack, and the other arguments are not used.
    """
    if capabilities is None:
        capabilities = chat_capabilities(
            web_search=web_search,
            workspace_root=workspace_root,
            activity_log=activity_log,
        )
    return Agent[None, str](
        name="chat",
        # The type the type checker holds every run to.
        output_type=str,
        # Each attempt at a request that takes longer fails; most hosted
        # models honour it, the given stand-in models do not.
        model_settings=ModelSettings(timeout=120),
        capabilities=capabilities,
    )


# Earlier demos import this name; the console chat builds its agent in build_app.
chat_agent = build_chat_agent(
    web_search=True, workspace_root=project_root() / "work" / "tinyjobs"
)
