"""Open the agent in a terminal: read the settings, build the app, run it.

From the chat lesson (02-04); since the settings lesson (03-01) it reads
no value and builds no object itself. Kept working: a later lesson that
changes the agent updates this module, or removes it and says why.
"""

from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from pydantic_tutor.composition import build_app
from pydantic_tutor.settings import Settings


def main() -> None:
    """Build the app from the settings, then open the chat."""
    try:
        app = build_app(settings=Settings())
    except (ValidationError, UserError) as error:
        # A setting or a key is wrong: say which, and stop, without a traceback.
        raise SystemExit(f"pydantic-tutor cannot start:\n{error}") from None
    app.agent.to_cli_sync(model=app.model, deps=app.deps, prog_name="pydantic-tutor")


if __name__ == "__main__":
    main()
