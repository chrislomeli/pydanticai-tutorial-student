"""The agent the chat talks to.

From the console-chat lesson (02-02); web search from the web-search
lesson (02-05); built from a setting in the settings lesson (03-01); a
workspace from the workspace lesson (03-03); the file tools from the file
tools lesson (03-04); the request log from the hooks lesson (04-02); the
batch limit from the capability lesson (04-03). No model here, as for
every agent in this project: the chat passes one on each run.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
    LocalWorkspace,
    Toolset,
    WebSearch,
)
from pydantic_ai_harness import FileSystem, Shell

from pydantic_tutor.capabilities.batch_limit import BatchLimit
from pydantic_tutor.capabilities.request_log import request_log
from pydantic_tutor.tools.workspace_tools import build_workspace_toolset

# Files the file tools may neither read nor write, in any folder: the
# library's own secret patterns, which by default it only keeps from being
# written. A leading `**/` means "in any folder, the top one included".
SECRET_PATTERNS = ("**/.env", "**/.env.*", "*.pem", "*.key", "**/secrets*")

# The only commands the shell may start: the chat runs the workspace's tests
# and reads files with the file tools. A guard against mistakes, not a
# boundary: only a command's first word is checked, and pytest runs any
# code it is given.
SHELL_COMMANDS = ("pytest",)


def build_chat_agent(*, web_search: bool, workspace_root: Path) -> Agent[None, str]:
    """Build the chat's agent, its runs working in the `workspace_root` folder."""
    capabilities: list[AbstractCapability[None]] = [
        # A place for tools to work in. It adds no tool: they use `ctx.workspace`.
        LocalWorkspace(working_dir=workspace_root),
        # Eight file tools, working in that workspace and kept to its folder.
        # The default `read_only_patterns` still stop writes; this stops reads.
        FileSystem(denied_patterns=SECRET_PATTERNS),
        # Four shell tools, running commands in that workspace, on this
        # machine. They keep to neither its folder nor SECRET_PATTERNS. Only
        # the allowed commands start; each is stopped after 30 seconds, the
        # default, or after the time the model asks for.
        Shell(allowed_commands=SHELL_COMMANDS),
        # Our two tools that read the workspace's Python modules, with their
        # instructions and time limit. Built with the folder they read.
        Toolset(toolset=build_workspace_toolset(root=workspace_root)),
    ]
    if web_search:
        # The model may search when it needs to. Each search is billed by
        # the provider, on top of the tokens.
        capabilities.append(WebSearch())
    # At most three tool calls of each response run: a listing and two
    # reads, not a runaway. The model can ask again for the rest.
    capabilities.append(BatchLimit(max_tool_calls=3))
    # Logs each request. Last, so it logs the request as every other
    # capability leaves it. It shows nothing until a program turns its
    # logger on; the chat does not.
    capabilities.append(request_log)
    return Agent(
        name="chat",
        instructions="You are a helpful assistant. Answer briefly.",
        capabilities=capabilities,
    )


# The agent with web search on and the default folder, as before: earlier
# scripts and the tests import it by name. The chat itself gets its agent
# from `build_app`.
chat_agent = build_chat_agent(web_search=True, workspace_root=Path("work/tinyjobs"))
