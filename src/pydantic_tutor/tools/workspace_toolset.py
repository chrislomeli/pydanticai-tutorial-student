"""The workspace's tools, with their instructions and time limit, as one toolset.

Built once, where the app is built, with the
folder the tools read: they take no deps.
"""

from pathlib import Path

from pydantic_ai.toolsets import FunctionToolset

from tutor_given.module_tools import ModuleTools


def build_workspace_toolset(*, root: Path) -> FunctionToolset[None]:
    """The tools that read the modules under `root/src`, with their instructions."""
    tools = ModuleTools(root=root)
    return FunctionToolset(
        tools=[tools.list_modules, tools.read_docstring],
        # Sent with every request, beside the agent's own instructions.
        instructions=f"Your workspace holds the Python project {root.resolve().name}.",
        # Set here: an agent's `tool_timeout` covers only its own `tools=`.
        timeout=10,
    )
