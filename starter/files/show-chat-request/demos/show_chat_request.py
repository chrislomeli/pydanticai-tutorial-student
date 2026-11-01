"""Show what the chat's agent sends the model: its tools and its instructions.

Given, from the workspace lesson's starter files (03-03), and first used in the
prompt lesson (04-04): you run it, you don't write it. Needs no API key: the
model is a stand-in that prints what it receives. Run `demos/fresh_copy.py` first.
"""

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import reports_what_it_is_sent

agent = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
agent.run_sync(user_prompt="hi", model=reports_what_it_is_sent())
