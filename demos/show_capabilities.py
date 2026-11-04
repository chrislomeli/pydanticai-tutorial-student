"""List the chat's capabilities, and what each one brings to the agent.

Given: you run it, you don't write it. Needs no API key; it makes no request.
Run `demos/fresh_copy.py` first.
"""

from pydantic_ai.capabilities import AbstractCapability

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings

# Two capabilities the library adds to every agent, for its own use.
ADDED_BY_THE_LIBRARY = {"ToolSearch", "PendingMessageDrainCapability"}
CONTRIBUTIONS = {
    "get_instructions": "instructions",
    "get_toolset": "tools",
    "get_workspace": "a workspace",
}


def overrides(*, kind: type[AbstractCapability[None]], method: str) -> bool:
    """Whether this kind of capability replaces the base class's method."""
    return getattr(kind, method) is not getattr(AbstractCapability, method)


def brings(*, kind: type[AbstractCapability[None]]) -> list[str]:
    """What this kind of capability contributes, in plain words."""
    found = [
        word
        for method, word in CONTRIBUTIONS.items()
        if overrides(kind=kind, method=method)
    ]
    hooks = [
        name
        for name in dir(AbstractCapability)
        if name.startswith(("before_", "after_", "wrap_", "on_"))
    ]
    if any(overrides(kind=kind, method=name) for name in hooks):
        found.append("hooks")
    return found


agent = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
for capability in agent.root_capability.capabilities:
    kind = type(capability)
    if kind.__name__ not in ADDED_BY_THE_LIBRARY:
        print(f"{kind.__name__}: {', '.join(brings(kind=kind))}")
