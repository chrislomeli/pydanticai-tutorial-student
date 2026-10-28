"""Ask the chat's agent to read every module, on a model that asks for all at once.

From the capability lesson (04-03); run `demos/fresh_copy.py` first. Kept
working: a later lesson that changes the agent updates it, or says why not.
"""

from pydantic_ai.messages import ModelResponse

from pydantic_tutor.chat.agent import build_chat_agent
from pydantic_tutor.settings import Settings
from tutor_given.stand_ins import asks_for_every_module_at_once

chat = build_chat_agent(web_search=False, workspace_root=Settings().workspace_root)
result = chat.run_sync(
    user_prompt="Read every module and summarise each.",
    model=asks_for_every_module_at_once(),
)
print("the run's history, response by response:")
for message in result.all_messages():
    if isinstance(message, ModelResponse):
        names = [call.tool_name for call in message.tool_calls]
        print("  tool calls:", ", ".join(names) or "none, the answer")
print("output:", result.output)
print("usage:", result.usage)
