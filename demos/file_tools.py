"""Show what the chat's file tools read, write and refuse, in its workspace.

From the file tools lesson (03-04). Needs no API key. Run `demos/fresh_copy.py`
first. Kept working: a later lesson that changes the chat's file tools
updates this script, or removes it and says why.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import calls_tool_with_the_prompt

agent = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)

# Each call is made by a stand-in model, with the path as the prompt; it
# answers with the tool's result, or the reason it was refused.
CALLS: list[tuple[str, str, dict[str, str | int]]] = [
    ("read_file", "README.md", {"limit": 1}),
    ("read_file", ".env", {}),
    ("write_file", ".env", {"content": "TOKEN=mine\n"}),
    ("read_file", "../../pyproject.toml", {}),
]
for tool_name, path, other_arguments in CALLS:
    result = agent.run_sync(
        user_prompt=path,
        model=calls_tool_with_the_prompt(
            tool_name=tool_name, argument="path", other_arguments=other_arguments
        ),
    )
    print(f"{tool_name} {path}:")
    for line in result.output.splitlines():
        print(f"  {line}")
