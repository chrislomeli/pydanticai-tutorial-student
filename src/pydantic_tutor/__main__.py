"""Open the agent in a terminal, with console tracing behind a flag.

From the chat lesson (02-04). Kept working: a later lesson that changes
the agent updates this module, or removes it and says why.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_tutor.agent import agent
from pydantic_tutor.codebase_tools import CodebaseDeps

# Ours, not the library's: a string naming the provider and the model, read
# by `to_cli_sync` on each call. The key it needs comes from the
# environment -- the provider reads it, not this project.
DEFAULT_MODEL = "openai:gpt-5-mini"

# Ours: which env var turns console tracing on. Named after this project,
# not the library, the way the reference project names its own
# `SANDBOX_TRACE`.
TRACE_VAR = "PYDANTIC_TUTOR_TRACE"


def configure_tracing() -> None:
    """Turn on console tracing when `PYDANTIC_TUTOR_TRACE` is set.

    Spans print to this terminal only: `send_to_logfire=False` means
    nothing leaves the machine, and no Logfire account or OTel collector
    is needed.
    """
    if not os.environ.get(TRACE_VAR):
        return
    import logfire

    logfire.configure(
        send_to_logfire=False, console=logfire.ConsoleOptions(colors="auto")
    )
    logfire.instrument_pydantic_ai()
    print(f"tracing on ({TRACE_VAR} is set); spans print to this terminal")


def main() -> None:
    """Load `.env`, configure tracing, then open the chat."""
    load_dotenv()
    configure_tracing()
    agent.to_cli_sync(
        model=DEFAULT_MODEL,
        deps=CodebaseDeps(root=Path("sample_code/tinyjobs")),
        prog_name="pydantic-tutor",
    )


if __name__ == "__main__":
    main()
