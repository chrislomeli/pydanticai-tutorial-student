"""A context gauge: how full each request's context was, and what a turn sent.

The measuring is the harness's (`estimate_context_tokens`,
`resolve_context_window`); this module is ours: one reading per request,
in a hook, and one line for the chat.
"""

from dataclasses import dataclass, field

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    ContextUsage,
    estimate_context_tokens,
    resolve_context_window,
)


@dataclass
class ContextGauge(AbstractCapability[None]):
    """One reading per request of the run it is given to: give each run a new one.

    A hook, not a listener for `ContextUsageEvent`: a listener changes how
    the run's requests are sent.
    """

    readings: list[ContextUsage] = field(default_factory=list)

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost: it reads each request as every other capability has
        # left it, which is what the model is sent.
        return CapabilityOrdering(position="innermost")

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Read the request about to be sent, and pass it on unchanged."""
        used = estimate_context_tokens(
            messages=request_context.messages,
            model_request_parameters=request_context.model_request_parameters,
        )
        # None: the window isn't known; assume the default, and say so.
        window = resolve_context_window(model=request_context.model)
        self.readings.append(
            ContextUsage(
                used_tokens=used,
                window_tokens=DEFAULT_CONTEXT_WINDOW if window is None else window,
                resolved=window is not None,
            )
        )
        return request_context

    def status_line(self) -> str:
        """The last request's context, and what the run sent in all."""
        last = self.readings[-1]
        assumed = "" if last.resolved else ", window assumed"
        sent = sum(reading.used_tokens for reading in self.readings)
        count = len(self.readings)
        requests = "request" if count == 1 else "requests"
        return (
            f"[context {last.used_tokens:,} of {last.window_tokens:,} tokens "
            f"({last.fraction:.1%}{assumed}) · this turn sent {sent:,} tokens "
            f"in {count} {requests}]"
        )
