"""Fetch starter files from the pydantic-tutor library into your project.

    uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py --list
    uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py workspace

Run it from your project's root, the folder with `pyproject.toml`. Each
name is an item of the library: some files, laid out where they belong in
your project. The program writes them there and prints what it did. It uses
only the standard library and needs no install.

- A file already in your project that is the same as the library's is left
  alone.
- A file that differs is not replaced: the item stops, nothing is written,
  and the differing files are listed. `--force` replaces them. Files of
  yours that the item does not name are never touched.
- `--into DIR` runs it against another project folder.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SOURCE = "https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter"


@dataclass(frozen=True)
class Item:
    name: str
    about: str
    files: dict[str, str]  # project-relative path -> sha256 of its content
    includes: tuple[str, ...]
    note: str


class LibraryError(Exception):
    """The library or the request is wrong; the message says what to do."""


def read(*, source: str, relative: str) -> bytes:
    """Bytes of `relative` under `source`: a URL, or a folder (the built library, for tests)."""
    if source.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(
                url=f"{source}/{relative}", timeout=30
            ) as response:
                data: bytes = response.read()
                return data
        except urllib.error.URLError as error:
            raise LibraryError(
                f"could not fetch {source}/{relative}: {error}"
            ) from error
    return (Path(source) / relative).read_bytes()


def load_items(*, source: str) -> dict[str, Item]:
    manifest = json.loads(read(source=source, relative="manifest.json"))
    return {
        name: Item(
            name=name,
            about=entry["about"],
            files=entry["files"],
            includes=tuple(entry["includes"]),
            note=entry["note"],
        )
        for name, entry in manifest["items"].items()
    }


def expand(*, items: dict[str, Item], names: list[str]) -> list[Item]:
    """The named items and what they include, each once, in order."""
    ordered: dict[str, Item] = {}

    def visit(name: str) -> None:
        if name not in items:
            raise LibraryError(f"no item called {name!r}; --list shows the names")
        for included in items[name].includes:
            visit(included)
        ordered.setdefault(name, items[name])

    for name in names:
        visit(name)
    return list(ordered.values())


def safe(*, relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise LibraryError(f"the library names a path outside your project: {relative}")
    return path


def digest(*, data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(*, source: str, item: Item, relative: str) -> bytes:
    data = read(source=source, relative=f"files/{item.name}/{relative}")
    if digest(data=data) != item.files[relative]:
        raise LibraryError(
            f"{item.name}/{relative} does not match the library's manifest; try again"
        )
    return data


def install(*, source: str, item: Item, project: Path, force: bool) -> None:
    """Write one item's files into `project`; write nothing if a file of yours would be replaced."""
    plan: list[tuple[str, Path, bool | None]] = []  # exists is None: already the same
    for relative, wanted in sorted(item.files.items()):
        target = project.joinpath(*safe(relative=relative).parts)
        if target.exists() and digest(data=target.read_bytes()) == wanted:
            plan.append((relative, target, None))
        else:
            plan.append((relative, target, target.exists()))
    differing = [relative for relative, _, exists in plan if exists]
    if differing and not force:
        listed = "\n".join(f"    {relative}" for relative in differing)
        raise LibraryError(
            f"{item.name}: these files differ from the library's and were not replaced:\n{listed}\n"
            "  Nothing was written. Replace them with --force."
        )
    print(f"{item.name}: {item.about}")
    for relative, target, exists in plan:
        if exists is None:
            print(f"  same      {relative}")
            continue
        data = fetch(source=source, item=item, relative=relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"  {'replaced' if exists else 'wrote   '}  {relative}")
    if item.note:
        print(f"\n{item.note}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="items to fetch")
    parser.add_argument("--list", action="store_true", help="list the items")
    parser.add_argument(
        "--into", type=Path, default=Path(), help="the project folder (default: here)"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace files that differ"
    )
    parser.add_argument(
        "--source", default=SOURCE, help="the library's URL or folder (for tests)"
    )
    args = parser.parse_args()
    try:
        items = load_items(source=args.source)
        if args.list or not args.names:
            for name, item in sorted(items.items()):
                print(f"{name:<22} {item.about}")
            return 0
        if not (args.into / "pyproject.toml").is_file():
            raise LibraryError(
                f"{args.into.resolve()} has no pyproject.toml: run this from your project's root, or use --into"
            )
        for item in expand(items=items, names=args.names):
            install(source=args.source, item=item, project=args.into, force=args.force)
    except LibraryError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
