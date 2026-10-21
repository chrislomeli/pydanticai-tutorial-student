from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import FileSystem, Shell

from pydantic_tutor.batch_limit import BatchLimit
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


def repo_workspace(ctx: RunContext[RepoDeps]) -> LocalWorkspace[RepoDeps]:
    """Give each run a workspace on this machine, in the repository it reads."""
    # Called once per run, with that run's deps: the workspace follows the
    # repository the run is given. Its folder is where relative paths
    # start, not a boundary: `..` and absolute paths reach past it.
    # Keep it to building the capability from deps: no I/O, no state.
    return LocalWorkspace(working_dir=ctx.deps.root)


# No model here: each run says which model to use, so the same agent runs on
# a test model now and a real one later. No repository either: each run
# passes one, as `deps`, and gets a workspace there.
agent = Agent(
    name="repo_assistant",
    deps_type=RepoDeps,
    instructions="You answer questions about a Python code repository. Answer briefly.",
    # Our two repository tools, with their instructions and time limit.
    toolsets=[repo_toolset],
    capabilities=[
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
        # At most three tool calls of each response run: enough for a
        # listing and two reads, not for a runaway. The model is sent
        # their results, and can ask again for the rest.
        BatchLimit(max_tool_calls=3),
        # Logs each request. Last, so it logs the request as the agent's
        # other capabilities leave it. A run's own capabilities come after.
        request_log,
    ],
    # Each attempt at a request that takes longer fails; most hosted
    # models honour it, the scripted models in these lessons do not.
    model_settings=ModelSettings(timeout=60),
)
