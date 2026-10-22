from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AgentCapability, LocalWorkspace, Toolset
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import FileSystem, Shell, ToolGuardrail
from pydantic_ai_harness.memory import FileStore

from pydantic_tutor.approval import approver, ask_first, refuse_unattended
from pydantic_tutor.batch_limit import BatchLimit
from pydantic_tutor.listable_memory import ListableMemory
from pydantic_tutor.prompt_file import PromptFile
from pydantic_tutor.repo_tools import RepoDeps, repo_toolset
from pydantic_tutor.request_log import request_log

# Files the file tools may neither read nor write, at any depth: the
# library's own secret patterns, which by default it only keeps from being
# written. A leading `**/` means "in any folder, the top one included".
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The only commands the shell may start: the agent runs the repository's
# tests, and reads files with the file tools. A guard against mistakes, not
# a boundary: only a command's first word is checked, and pytest runs any
# code it is given.
SHELL_COMMANDS = ("pytest",)

# The agent's prompt, in a file of its own beside this module, so it is
# found whichever folder the program runs from.
PROMPT_PATH = Path(__file__).parent / "prompts" / "repo_assistant.md"


def repo_workspace(ctx: RunContext[RepoDeps]) -> LocalWorkspace[RepoDeps]:
    """Give each run a workspace on this machine, in the repository it reads."""
    # Called once per run, with that run's deps: the workspace follows the
    # repository the run is given. Its folder is where relative paths
    # start, not a boundary: `..` and absolute paths reach past it.
    # Keep it to building the capability from deps: no I/O, no state.
    return LocalWorkspace(working_dir=ctx.deps.root)


def repo_capabilities() -> list[AgentCapability[RepoDeps]]:
    """Everything the agent does, as a list of capabilities, each one new.

    The order is an arrangement, not a rule: a capability whose place
    matters declares it (`PromptFile` first, `request_log` last), so these
    may be listed, or added to, in any order.
    """
    return [
        # The agent's instructions, read from PROMPT_PATH when this list is
        # built. Text, so it is sent before the repository's name, which the
        # toolset computes for each request.
        PromptFile(path=PROMPT_PATH),
        # Our two repository tools, with their instructions and time limit.
        Toolset(toolset=repo_toolset),
        repo_workspace,
        # File tools in the run's workspace. They keep to its folder, and
        # refuse the secret files; the default `read_only_patterns` still
        # keeps `.git/` and the rest from being written.
        FileSystem(denied_patterns=SECRET_PATTERNS),
        # Commands in the run's workspace, on this machine. They keep to
        # neither its folder nor the file tools' SECRET_PATTERNS. Only the
        # allowed commands start; each is stopped after 30 seconds, the
        # default, or after the time the model asks for.
        Shell(allowed_commands=SHELL_COMMANDS),
        # Every call to an ASK_FIRST tool waits for approval before it runs.
        ToolGuardrail(guard=ask_first),
        # The answer when no one is there: each held call is refused, and
        # the model told why. A run replaces it with its own `approver(...)`.
        approver(answer=refuse_unattended),
        # A notebook the model keeps across sessions, as Markdown files in
        # `.agent-memory/` in the run's workspace: each repository has its
        # own. Injection off: no request carries the notebook; the model
        # lists and reads it with tools when it chooses to.
        ListableMemory(store=FileStore(directory=".agent-memory"), inject_memory=False),
        # At most three tool calls of each response run: enough for a
        # listing and two reads, not for a runaway. The model is sent
        # their results, and can ask again for the rest.
        BatchLimit(max_tool_calls=3),
        # Logs each request, as every other capability leaves it.
        request_log,
    ]


def build_agent(
    *, capabilities: Sequence[AgentCapability[RepoDeps]] | None = None
) -> Agent[RepoDeps, str]:
    """Build the repository assistant: what it is, here; what it does, in `capabilities`.

    The constructor sets what no capability can, the agent's name and the
    type of its deps and of its output, and its defaults for every
    request, which are configuration, not behaviour, though a capability
    could carry them. `capabilities` defaults to `repo_capabilities()`; pass
    another list to build the same agent with its stack rearranged or
    extended.
    """
    return Agent(
        name="repo_assistant",
        # The types the type checker holds every run and tool to.
        deps_type=RepoDeps,
        output_type=str,
        # Each attempt at a request that takes longer fails; most hosted
        # models honour it, the scripted models in these lessons do not.
        model_settings=ModelSettings(timeout=60),
        capabilities=repo_capabilities() if capabilities is None else capabilities,
    )


# The project's one agent, used by the app and the scripts. No model here:
# each run says which model to use, so the same agent runs on a test model
# now and a real one later. No repository either: each run passes one, as
# `deps`, and gets a workspace there.
agent = build_agent()
