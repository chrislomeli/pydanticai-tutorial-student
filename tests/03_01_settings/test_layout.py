"""The layout's boundaries: what may import what.

`capabilities/`, `stores/` and `tools/` know the library and each other. They
never import the chat, the settings or the composition root, so any of them
can move into a package of its own.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "pydantic_tutor"
LIBRARIES = ("capabilities", "stores", "tools")
APPLICATION = (
    "pydantic_tutor.chat",
    "pydantic_tutor.composition",
    "pydantic_tutor.settings",
)


def imported(*, path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_capabilities_stores_and_tools_do_not_import_the_application() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE)} imports {name}"
        for folder in LIBRARIES
        for path in sorted((PACKAGE / folder).glob("*.py"))
        for name in sorted(imported(path=path))
        if name.startswith(APPLICATION)
    ]
    assert offenders == []
