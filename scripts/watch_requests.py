"""Run the agent once on TestModel, with its request log turned on.

From the hooks lesson (04-02). Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

import logging
import sys
from pathlib import Path

from pydantic_ai.models.test import TestModel

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps

# The program that runs the agent decides which logs are shown, and where.
# Here: each line as its logger's name and its message, on stdout, so it
# prints in order with this script's own lines; and our request log on.
logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
logging.getLogger(name="pydantic_tutor.request_log").setLevel(level=logging.INFO)

# TestModel calls list_modules, then answers: a run of two requests.
result = agent.run_sync(
    user_prompt="What does this codebase do?",
    model=TestModel(call_tools=["list_modules"]),
    deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
)
print("output:", result.output)
print("usage:", result.usage)
