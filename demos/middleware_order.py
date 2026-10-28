"""Hooks run before the request in list order, and after it in reverse.

From the capability lesson (04-03); run `demos/fresh_copy.py` first. Kept
working: a later lesson that changes `BatchLimit` updates it, or says why not.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse

from pydantic_tutor.capabilities.batch_limit import BatchLimit
from pydantic_tutor.settings import Settings
from pydantic_tutor.tools.workspace_tools import build_workspace_toolset
from tutor_given.stand_ins import asks_for_five_reads


@dataclass(kw_only=True)
class Watch(AbstractCapability[None]):
    """Print each request going out and each response coming back, unchanged."""

    # Printed at the start of each line, to tell two of them apart.
    name: str

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Say that a request is about to go out, and pass it on."""
        print(f"{self.name}: before the request")
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Say how many tool calls the response holds, and pass it on."""
        print(
            f"{self.name}: after, the response has {len(response.tool_calls)} tool calls"
        )
        return response


reader: Agent[None, str] = Agent(
    name="reader",
    toolsets=[build_workspace_toolset(root=Settings().workspace_root)],
    capabilities=[
        Watch(name="first"),
        BatchLimit(max_tool_calls=3),
        Watch(name="last"),
    ],
)
reader.run_sync(
    user_prompt="Read every module and summarise each.", model=asks_for_five_reads()
)
