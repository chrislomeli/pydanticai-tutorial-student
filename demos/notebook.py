"""Two sessions with the chat's agent: one stores a note, the next finds it.

From the memory lesson (04-05). The notebook lives in the workspace, so run
`demos/fresh_copy.py` first. Kept working: a later lesson that changes
the agent updates this script, or removes it and says why.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import remembers_then_recalls

chat = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
model = remembers_then_recalls()
for prompt in ["Remember that we log to stderr.", "What did we decide about logging?"]:
    # No message_history: a new session, as far as the model can tell.
    print(f"session: {prompt}")
    result = chat.run_sync(user_prompt=prompt, model=model)
    print("  output", result.output)
