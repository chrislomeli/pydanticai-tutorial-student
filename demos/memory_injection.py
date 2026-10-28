"""What memory injection would send: the notebook beside every question.

From the memory lesson (04-05). A second agent reads the notebook
`demos/notebook.py` left in the workspace, with injection on. Kept
working: a later lesson updates it, or removes it and says why.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore

from pydantic_tutor.chat.agent import NOTEBOOK
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import prints_every_message

# The chat's notebook; only `inject_memory` differs.
injecting: Agent[None, str] = Agent(
    capabilities=[
        LocalWorkspace(working_dir=Settings().workspace_root),
        Memory(store=FileStore(directory=NOTEBOOK), inject_memory=True),
    ],
)
model = prints_every_message()
history: list[ModelMessage] = []
for prompt in ["What does tinyjobs do?", "And how is it tested?"]:
    print(f"turn: {prompt}")
    result = injecting.run_sync(
        user_prompt=prompt, model=model, message_history=history
    )
    history = result.all_messages()
