"""Show what the chat's shell runs and refuses, in its workspace.

From the shell lesson (03-05). Needs no API key. Run `demos/fresh_copy.py`
first: the commands run in `work/tinyjobs`. Kept working: a later lesson
that changes the chat's shell updates this script, or removes it and says
why.
"""

import textwrap

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Shell

from pydantic_tutor.capabilities.approval import approver
from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import approve_all, calls_tool_with_the_prompt

root = Settings().workspace_root
chat = build_chat_agent(web_search=False, workspace_root=root)

# A stand-in model runs each line as a command and answers with what came back.
COMMANDS = [
    "pytest -q --tb=no",
    "rm README.md",
    "cat .env",
    "pytest --version && cat .env",
]
for line in COMMANDS:
    result = chat.run_sync(
        user_prompt=line,
        model=calls_tool_with_the_prompt(tool_name="run_command", argument="command"),
        # Every call approved: this script shows what the tools themselves
        # allow and refuse, once a person has said yes.
        capabilities=[approver(answer=approve_all)],
    )
    print(f"$ {line}")
    print(textwrap.indent(text=result.output.rstrip(), prefix="  "))

# The shell stops a command that runs too long. This agent allows `sleep` and a
# one-second limit, to show it without waiting thirty.
patient = Agent(
    capabilities=[
        LocalWorkspace(working_dir=root),
        Shell(allowed_commands=("sleep",), default_timeout=1.0),
    ]
)
result = patient.run_sync(
    user_prompt="sleep 5",
    model=calls_tool_with_the_prompt(tool_name="run_command", argument="command"),
)
print("$ sleep 5")
print(textwrap.indent(text=result.output.rstrip(), prefix="  "))
