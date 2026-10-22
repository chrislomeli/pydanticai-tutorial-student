"""Open the agent in a terminal: read the settings, build the app, open the chat.

Reads no value and builds no object itself: the settings and
`build_app` do. A wrong setting or a missing key stops it with a message
saying which, not a traceback, before the chat opens.
"""

from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from pydantic_tutor.chat import chat
from pydantic_tutor.composition import build_app
from pydantic_tutor.settings import Settings


def main() -> None:
    """Build the app from the settings, then open the chat."""
    try:
        app = build_app(settings=Settings())
    except (ValidationError, UserError) as error:
        # A setting or a key is wrong: say which, and stop, without a traceback.
        raise SystemExit(f"pydantic-tutor cannot start:\n{error}") from None
    chat(app=app)


if __name__ == "__main__":
    main()
