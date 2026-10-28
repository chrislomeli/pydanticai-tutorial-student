"""Run the agent outside the chat, on the model that asks which flag.

From the ask-user lesson (05-03). A script's run is not the chat's: no one is
there to answer, so the agent offers the model no question tool. Run
`demos/fresh_copy.py` first. Kept working by later lessons.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import asks_which_flag

settings = Settings()
agent = build_chat_agent(web_search=False, workspace_root=settings.workspace_root)
result = agent.run_sync(user_prompt="Add a flag to the CLI.", model=asks_which_flag())
print(result.output)
