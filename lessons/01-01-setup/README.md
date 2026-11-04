# Setting up the environment

## Welcome

Let's create the project you'll build in for the rest of the series.
You start from an empty folder, run a handful of commands, and write one
small file: check your tools, create a Python project with `uv`, install
Pydantic AI and its harness at pinned versions, run a five-line agent —
no API key, no cost — protect a secrets file you haven't made yet, add
the code checks, and commit the lot to git.

That commit is on your own machine; nothing needs a GitHub account. From
the next lesson on you can keep building on this project, or clone the
lesson starting points we publish — handy if you skip a lesson or yours
breaks. A starting point isn't always identical to what you built: some
lessons add files on it, such as their own tests. Both routes work, and
each lesson says which files it expects.

By the end, you'll be able to:

1. **Check your tools**: `git`, and `uv` 0.13.1 or newer.
2. **Create a `uv` project** pinned to `pydantic-ai` 2.56.0 and
   `pydantic-ai-harness` 0.56.0 on Python 3.14, say why those two pins
   must move together, and check its code with Ruff and mypy.
3. **Run an agent with no API key**, and make sure your `.env` file can't
   be committed by accident.

About 15 minutes, not counting installing tools. The series is tested with
Python 3.14, `uv` 0.13.1, Ruff 0.16.10 and mypy 2.4.0.

## What you'll build

You start from an empty folder and end with a Python project, in its own
git repository, with a first agent in it. Almost every file is made by a
command you run. You write one file yourself, add an empty one, and add a few
lines to two others:

```text
pydantic-tutor/
├── .gitignore              made by uv; you add one line for .env
├── .python-version         made by uv: 3.14
├── README.md               made by uv, empty
├── hello_agent.py          you write it: a five-line agent that needs no API key
├── pyproject.toml          made by uv: the project, its dependencies and their pins;
│                           you add the code checks' settings
├── uv.lock                 made by uv: the exact version of every package installed
└── src/
    └── pydantic_tutor/
        ├── __init__.py     made by uv: the package, with a hello-world command
        └── py.typed        you add it, empty: the package's types can be trusted
```

There's nothing to fetch first: this lesson starts from an empty folder.
The steps, in order:

1. **Check your tools**: `git` and `uv`, and that git knows your name.
2. **Create the project** with `uv init`, and run its command.
3. **Install Pydantic AI** and the harness, at pinned versions.
4. **Run your first agent**: five lines of Python on `TestModel`.
5. **Keep secrets out of git**: make sure `.env` can't be committed by
   accident.
6. **Check your code**: add Ruff and mypy, and run them until they report
   nothing.
7. **Commit.**

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Step 1: Check your tools

> **Goal:** confirm you have `git`, and `uv` 0.13.1 or newer, and that git
> knows your name and email.

Everything in this lesson runs in a terminal, on macOS or Linux, with two
tools: `git` and `uv`. You don't need to install Python: `uv` downloads
the version the project asks for, the first time it needs it.

Check that both tools are there:

```bash
git --version
uv --version
```

If either command isn't found, or `uv` is older than 0.13.1, follow the
guide for your system, then come back:

- [macOS](../00-04-macos-installs/)
- [Linux](../00-05-linux-installation/)
- [Windows](../00-06-windows-installation/): you work inside WSL, a Linux system on
  Windows

Then check that git knows who you are:

```bash
git config user.name
git config user.email
```

If either prints nothing, set it:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

> [!TIP]
> **Check:** the first two commands print versions, such as:
>
> ```text
> git version 2.50.1 (Apple Git-155)
> uv 0.13.1 (62faed335 2026-10-13 aarch64-apple-darwin)
> ```
>
> `uv` is 0.13.1 or newer, and the two `git config` commands each print a
> value.

**What it means**

Your versions may be newer than these; that's fine. Your name and email
matter because two things use them. `uv` copies them into the project's
`pyproject.toml` in the next step, and every commit records who made it.
On some systems a commit fails without them. On others git guesses, and
the guess ends up in your history.

## Step 2: Create the project

> **Goal:** create a `uv` project for Python 3.14, and run the command it
> comes with.

This makes the project every later lesson adds to. It comes first
because every other command runs inside it.

Make an empty folder for the project, outside any other git repository,
and go into it. Every command from here on runs inside this folder.

```bash
mkdir pydantic-tutor
cd pydantic-tutor
uv init --package --python 3.14
```

You should see:

```text
Initialized project `pydantic-tutor`
```

List what's there now:

```bash
ls -A
```

```text
.git
.gitignore
.python-version
README.md
pyproject.toml
src
```

The order can differ on your system, which sorts names its own way; the
six names must all be there.

> [!TIP]
> **Check:** run the command the project comes with:
>
> ```bash
> uv run pydantic-tutor
> ```
>
> You should see this. Your paths and timings will differ, and so may
> Python's patch version (3.14.8 here). If `uv` already has a Python 3.14,
> the two download lines are missing:
>
> ```text
> Downloading cpython-3.14.8-macos-aarch64-none (download) (25.4MiB)
>  Downloaded cpython-3.14.8-macos-aarch64-none (download)
> Using CPython 3.14.8
> Creating virtual environment at: .venv
>    Building pydantic-tutor @ file:///Users/you/pydantic-tutor
>       Built pydantic-tutor @ file:///Users/you/pydantic-tutor
> Installed 1 package in 1ms
> Hello from pydantic-tutor!
> ```

**What it means**

The last line is a small hello-world that `uv init` wrote for you: a
`main()` function in `src/pydantic_tutor/__init__.py`, which
`pyproject.toml` turns into the `pydantic-tutor` command you ran.

Everything above it is `uv`'s one habit: sync, then run. Every `uv run`
first makes the environment match the project, then runs your command.
Before running yours, `uv` downloaded Python 3.14 (only the first time),
created `.venv` — the project's private virtual environment, where its
packages live — and installed your project into it. `ls -A` showed no
`.venv` a moment ago. It exists now because you ran `uv run`.

The two flags shaped what you got. `--package` makes the project
installable, with its code under `src/`; that's already `uv`'s default at
0.13.1, so passing it makes the layout explicit in the command itself.
`--python 3.14` wrote `3.14` to `.python-version`, which `uv` reads to
choose this project's Python. It also added `requires-python = ">=3.14"`
to `pyproject.toml`: the project runs on 3.14 here, and declares that it
needs 3.14 or newer wherever it runs next. `uv` also filled in your name
and email there, from your git settings.

The folder wasn't a git repository yet, so `uv init` made it one
(`.git`) and wrote a `.gitignore` along with it. That's why the folder
had to be outside any other repository: inside one, `uv init` skips both,
and steps 5 and 7 won't work as written.

## Step 3: Install Pydantic AI and the harness

> **Goal:** add the two packages the series is built on, at pinned
> versions, and see the third one they bring in.

The project runs, but it has no Pydantic AI in it yet. One command adds
it, at the versions this lesson was tested with:

```bash
uv add "pydantic-ai==2.56.0" "pydantic-ai-harness==0.56.0"
```

`uv` prints a few lines of progress (the first time, it names the larger
packages as it downloads them). Then it prints one line for every
package it installed: about a hundred, each starting with `+`, in
alphabetical order. Your two, `+ pydantic-ai==2.56.0` and
`+ pydantic-ai-harness==0.56.0`, are somewhere in the middle. You don't
need to read the list. The check below shows what matters.

Why two pins? `pydantic-ai` installs the framework through
`pydantic-ai-slim`, along with optional integrations.
`pydantic-ai-harness` adds capabilities such as access to files and a
shell. For these releases, both depend on the same exact version of
`pydantic-ai-slim`: 2.56.0. That's why we pin them together. You don't
need to remember the packaging details; the check below lets you see
the shared dependency for yourself.

Open `pyproject.toml`. Yours should look like this, with your own name
and email:

`pyproject.toml`:

```toml
[project]
name = "pydantic-tutor"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "Ada Student", email = "ada@example.com" }
]
requires-python = ">=3.14"
dependencies = [
    "pydantic-ai==2.56.0",
    "pydantic-ai-harness==0.56.0",
]

[project.scripts]
pydantic-tutor = "pydantic_tutor:main"

[build-system]
requires = ["uv_build>=0.13.1,<0.14.0"]
build-backend = "uv_build"
```

Before you run the check, predict: you asked for two packages. Which
package do you think both of them depend on?

> [!TIP]
> **Check:** ask `uv` what depends on `pydantic-ai-slim`:
>
> ```bash
> uv tree --invert --package pydantic-ai-slim --depth 1
> ```
>
> You should see this. Check that `pydantic-ai-slim` says `v2.56.0` and
> `pydantic-ai-harness` says `v0.56.0`. The number of packages resolved
> can differ: `uv` picks the newest releases of the packages your pins
> don't fix, and those change over time:
>
> ```text
> Resolved 107 packages in 2ms
> pydantic-ai-slim v2.56.0
> ├── pydantic-ai[anthropic, cli, evals, google, logfire, mcp, openai, web] v2.56.0
> ├── pydantic-ai-harness v0.56.0
> └── pydantic-evals v2.56.0
> ```

**What it means**

There's the shared dependency: `pydantic-ai-slim` at 2.56.0. The
square brackets on the `pydantic-ai` line list optional integrations;
`pydantic-evals` is another dependency in the tree. You don't need to
study those packages now. The important point is that your two pins
resolve to a compatible framework version. When you upgrade later,
check that the versions still agree.

Notice that you didn't edit `pyproject.toml` at all: `uv init` and
`uv add` wrote all of it between them. Keep adding and removing
dependencies through `uv`, never by hand, so the file always matches
what's installed.

## Step 4: Run your first agent

> **Goal:** run an agent with no API key, and read what Pydantic AI's
> banner tells you.

This first agent is deliberately small: it creates an `Agent` with
`TestModel`, runs it on a greeting, and prints the result. **Notice
`TestModel()`**: that's why this first run needs neither an API key nor
a paid model. You don't need to understand the rest of the agent yet;
the next lesson will unpack it.

Create a file called `hello_agent.py` in the project folder, next to
`pyproject.toml`, with this content:

`hello_agent.py`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(model=TestModel())
result = agent.run_sync(user_prompt="Hello!")
print(result.output)
```

> [!TIP]
> **Check:** run it:
>
> ```bash
> uv run hello_agent.py
> ```
>
> You should see this. The banner's layout depends on how wide your
> terminal is: in a narrower one, its lines wrap differently, and below
> about 60 columns the drawing is left out. Check the first line's
> versions and the last line:
>
> ```text
>                  pydantic-ai v2.56.0 • pydantic-ai-harness v0.56.0 • Python 3.14.8
>       / \
>      /   \       agent: agent • model: test:test • tools: 0 • capabilities: 0
>    /___.___\
>   /    |    \    observability: off — see every model and tool call live, with cost
> /      |      \    set it up free with Logfire and a GitHub login: https://pydantic.dev/ai-setup.md
> `---.._|_..---'    or use any OpenTelemetry backend: https://pydantic.dev/docs/ai/logfire/#otel
>
>                  goes away once observability is on — or PYDANTIC_AI_NO_BANNER=1
> success (no tool calls)
> ```

**What it means**

The last line is your first agent answering. `run_sync` ran the agent
and waited for the result, and `result.output` is the answer. It came
from `TestModel`, a stand-in model that ships with Pydantic AI: it
answers at once, inside your own program, calling no provider, so it
needs no API key and costs nothing. Unless you tell it what to say, it
replies `success (no tool calls)` whenever no tool was called.

Everything above that line is the banner. Pydantic AI prints it once
per process, the first time an agent runs. It confirms the versions you
installed, the agent's name (taken from the variable you assigned it to,
`agent`) and the model it used. It also points you toward observability,
a way to watch every model and tool call, which a later lesson covers.

The banner is only there to confirm what you already know. Guess what
the output looks like with it turned off, then set
`PYDANTIC_AI_NO_BANNER` and check your guess:

```bash
PYDANTIC_AI_NO_BANNER=1 uv run hello_agent.py
```

```text
success (no tool calls)
```

Notice the code itself: `Agent(model=...)`, `run_sync(user_prompt=...)`.
Every argument has its name. Python would accept them without the names,
but this series passes arguments by keyword everywhere, so every call
says plainly what it's doing.

Keep `hello_agent.py` after this lesson. Whenever your environment looks
wrong later on, running it again tells you in a second whether Pydantic
AI still works.

## Step 5: Keep secrets out of git

> **Goal:** make sure a `.env` file can't be committed by accident, before
> it exists.

You don't create `.env` yet. You make sure git will never pick it up by
accident, because the only reliable time is before the file exists: from
the chat lesson on, `.env` will hold your API key, and a key that reaches
git history can be used by anyone with a copy of the repository — it
stays there even if you delete the file in a later commit.

Ask git whether it would ignore a `.env` file now:

```bash
git check-ignore -v .env
```

It prints nothing: nothing in `.gitignore` covers `.env` yet.

Open `.gitignore` and add these lines at the end, starting with the blank
line:

`.gitignore` (add at the end):

```gitignore

# Secrets: holds your API key from the chat lesson on. Never commit it.
.env
.env.*
```

The whole file should now read:

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

# Secrets: holds your API key from the chat lesson on. Never commit it.
.env
.env.*
```

> [!TIP]
> **Check:** ask git again:
>
> ```bash
> git check-ignore -v .env
> ```
>
> You should see:
>
> ```text
> .gitignore:13:.env	.env
> ```

**What it means**

Git named the rule that matched: line 13 of `.gitignore`, the line you
added. And there's still no `.env` in the folder. Git checks a name
against `.gitignore`, not a file, so `git add` will skip `.env` from the
day you create it. That stops accidents, not decisions: `git add -f
.env` adds it anyway, so never use `-f` on a file that holds a secret.

The rest of the file is what `uv init` wrote. It already ignored
Python's caches, build files and `.venv`, but it said nothing about
secrets until you added that line.

## Step 6: Check your code

> **Goal:** add Ruff and mypy to the project, and run them until they
> report nothing.

The code runs. Now you add the two checkers from the tools lesson —
Ruff, which formats and lints, and mypy, which checks every use of a
value against its type — so the code is also written to a standard
before the first commit.

Add them to the project as development tools, with a third beside them,
pytest, the test runner, at the versions this lesson was tested with:

```bash
uv add --dev "ruff==0.16.10" "mypy==2.4.0" "pytest==9.1.1"
```

`uv` lists what it installed, in lines starting with `+`: `ruff`, `mypy`,
`pytest`, and six small packages they need. One line starts with `~`:
`uv` rebuilt your own project, `pydantic-tutor`.

The commands still need a little configuration. Add the following
settings to `pyproject.toml` so the checks run consistently in later
lessons. You don't need to study the individual settings now.

Add these lines at the end of the file, starting with the blank line:

`pyproject.toml` (add at the end):

```toml

[tool.ruff.lint]
select = [
    "E", "W",  # style: pycodestyle
    "F",       # unused imports and names, undefined names
    "I",       # import order
    "N",       # PEP 8 names
    "UP",      # modern syntax for this Python
    "B",       # likely bugs
    "ANN",     # every parameter and return annotated; ANN401 flags Any
    "FBT",     # no booleans by position, in a call or a def
    "SIM",     # simpler equivalents
    "RUF",     # Ruff's own rules
]
ignore = ["E501"]  # line length is the formatter's job

[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]
exclude = ["^tests/"]  # the series' lesson tests refer to code you have not written yet
```

The whole file should now read, with your own name and email:

```toml
[project]
name = "pydantic-tutor"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "Ada Student", email = "ada@example.com" }
]
requires-python = ">=3.14"
dependencies = [
    "pydantic-ai==2.56.0",
    "pydantic-ai-harness==0.56.0",
]

[project.scripts]
pydantic-tutor = "pydantic_tutor:main"

[build-system]
requires = ["uv_build>=0.13.1,<0.14.0"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "mypy==2.4.0",
    "pytest==9.1.1",
    "ruff==0.16.10",
]

[tool.ruff.lint]
select = [
    "E", "W",  # style: pycodestyle
    "F",       # unused imports and names, undefined names
    "I",       # import order
    "N",       # PEP 8 names
    "UP",      # modern syntax for this Python
    "B",       # likely bugs
    "ANN",     # every parameter and return annotated; ANN401 flags Any
    "FBT",     # no booleans by position, in a call or a def
    "SIM",     # simpler equivalents
    "RUF",     # Ruff's own rules
]
ignore = ["E501"]  # line length is the formatter's job

[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]
exclude = ["^tests/"]  # the series' lesson tests refer to code you have not written yet
```

One more file, and it stays empty. Create `src/pydantic_tutor/py.typed`
beside `__init__.py`, with nothing in it.

It's a marker, defined by [PEP 561](https://peps.python.org/pep-0561/):
it tells type checkers that your package's annotations can be trusted.
`mypy .` doesn't need it, because it reads the package as part of the
project. But check one script on its own, `uv run mypy some_script.py`,
and mypy finds `pydantic_tutor` as an installed package. Without the
marker, it refuses to read its types and reports `module is installed,
but missing library stubs or py.typed marker`.

Last, let Ruff lay out your code in its style. Run this whenever you've
written or pasted code:

```bash
uv run ruff format
```

It says how many files it reformatted and how many it left unchanged;
the numbers depend on your files. This is the first `uv` command since
you edited `pyproject.toml`, so `uv` first rebuilds the project and
prints a few lines of its own (`Building pydantic-tutor`,
`Built pydantic-tutor`, `Uninstalled 1 package`, `Installed 1 package`).

> [!TIP]
> **Check:** run the two checks:
>
> ```bash
> uv run ruff check
> uv run mypy .
> ```
>
> You should see:
>
> ```text
> All checks passed!
> Success: no issues found in 2 source files
> ```

**What it means**

Clean, which is the normal state from here on, and the habit that keeps
it that way: format, then check, before every commit. When a check finds
something, fix it; a comment such as `# noqa` or `# type: ignore` hides
a finding instead, so save it for the rare case where there's truly no
better way. The three commands ran in a deliberate order. `ruff format` went first, so `ruff check` has nothing
left to say about layout. Ruff looked at your two Python files and
`README.md`, since it also formats any Python code blocks it finds inside
Markdown. `mypy .` checked the project's two Python files.

You don't see a finding today, so here's one. Before you look, guess: a
file has `def greet(name):` with no types at all, and calls it as
`greet("hello")`. What would mypy say about it? Here's what it says:

```text
bad.py:3: error: Function is missing a type annotation  [no-untyped-def]
bad.py:6: error: Call to untyped function "greet" in typed context  [no-untyped-call]
Found 2 errors in 1 file (checked 1 source file)
```

Every finding names the file, the line and the rule, which makes it easy
to find and fix. Both of these come from `strict = true`, which makes
mypy treat a missing annotation, and a call to a function that has none,
as an error rather than a pass.

The rest of the settings you added work the same way. Ruff's `select`
list turns on rule sets by their code. A few hold you to the most: `ANN`
means every parameter and return value is annotated, with `ANN401`
flagging `Any`; `FBT` means no `True` or `False` passed by position, and
no `bool` parameter that could be; `N` is Python's naming conventions.
The `pydantic.mypy` plugin teaches mypy how Pydantic models are built.

The `--dev` flag made these three a **development dependency**: a package
you need to work on the project, but not to run it. `uv` records them
under `[dependency-groups]` rather than `dependencies`, and `uv run`
installs both. This lesson writes no tests, but pytest is there for the
tests the project ends up holding, run with `uv run pytest`.

This is the one time in the lesson you edited `pyproject.toml` by hand.
The dependencies are still `uv`'s to write. The `[tool.…]` tables are
settings `uv` never touches, so they're yours.

## Step 7: Commit

> **Goal:** commit the project, and say why `.venv` isn't in it.

The project is complete and clean. Save it in git, so every later lesson
starts from here.

See what you're about to commit:

```bash
git status --short --untracked-files=all
```

```text
?? .gitignore
?? .python-version
?? README.md
?? hello_agent.py
?? pyproject.toml
?? src/pydantic_tutor/__init__.py
?? src/pydantic_tutor/py.typed
?? uv.lock
```

Commit them:

```bash
git add -A
git commit -m "Set up the project"
```

> [!TIP]
> **Check:** you should see:
>
> ```text
> [main (root-commit) 84c723c] Set up the project
>  8 files changed, 2294 insertions(+)
>  create mode 100644 .gitignore
>  create mode 100644 .python-version
>  create mode 100644 README.md
>  create mode 100644 hello_agent.py
>  create mode 100644 pyproject.toml
>  create mode 100644 src/pydantic_tutor/__init__.py
>  create mode 100644 src/pydantic_tutor/py.typed
>  create mode 100644 uv.lock
> ```
>
> Your commit id will differ, and your branch may be called `master`
> rather than `main`, depending on your git settings. The number of
> insertions can differ too, with the size of `uv.lock`. None of these
> matters here.

**What it means**

Eight files, and the biggest folder in the project, `.venv`, isn't among
them: `.gitignore` leaves it out, and `uv.lock` goes in, so a copy of the
project can rebuild `.venv` with the same versions you have. `uv` wrote `uv.lock` at your first
`uv run` and updated it each time you added packages.

There's no `.ruff_cache` or `.mypy_cache` in the list either. Both tools
keep a folder like that to run faster next time, and each one holds a
`.gitignore` of its own that ignores everything inside it.

## What you can now do

That's the project, set up. You know what it asked for, what
that resolved to and what's installed; why the framework's two pins move
together; that an agent can run with no key; and that your code and your
secrets are checked before anything is committed. Here's what you hold
now:

| File or command | What it's for |
|---|---|
| `pyproject.toml` | What you asked for: the project, its pinned dependencies, and the checks' settings |
| `uv.lock` | What that resolved to: one exact version of every package, for every system |
| `.venv` | What's installed. Every `uv run` makes it match the lockfile first, so it's never committed |
| `pydantic-ai-slim` | The framework you import. `pydantic-ai` and the harness both require it at one exact version, so they move together |
| `uv run ruff format`, `ruff check`, `mypy .` | Your code checks, run before every commit |
| `hello_agent.py` | A one-second check that Pydantic AI works, with no API key |
| `.env` in `.gitignore` | Your key stays out of git, from the day you create it |

## Docs vs reality

The Pydantic AI
[installation page](https://pydantic.dev/docs/ai/overview/install/)
covers `pydantic-ai` and `pydantic-ai-slim`, and says they need Python
3.11 or newer, which matches the 2.56.0 release. It doesn't mention
`pydantic-ai-harness`, that the harness pins the library to one exact
version, or the banner. When you upgrade this project later, upgrade the
two packages together.

## Going further

Nothing here is needed for the next lesson. Each item is for when you're
curious.

- **The commands, as a checklist.** For when you set up the project
  again, or on another machine. It's a checklist, not the whole recipe:
  the steps above have the file contents and what each command should
  print. Run it from the folder that will hold `pydantic-tutor`, with
  `uv` 0.13.1 or newer:

  ```bash
  mkdir pydantic-tutor
  cd pydantic-tutor
  uv init --package --python 3.14
  uv run pydantic-tutor
  uv add "pydantic-ai==2.56.0" "pydantic-ai-harness==0.56.0"
  uv tree --invert --package pydantic-ai-slim --depth 1
  # write hello_agent.py (step 4)
  uv run hello_agent.py
  # add .env to .gitignore (step 5)
  git check-ignore -v .env
  uv add --dev "ruff==0.16.10" "mypy==2.4.0" "pytest==9.1.1"
  # add the checks' settings to pyproject.toml, and an empty
  # src/pydantic_tutor/py.typed (step 6)
  uv run ruff format
  uv run ruff check
  uv run mypy .
  git add -A
  git commit -m "Set up the project"
  ```

- **Put your project on GitHub (optional).** No lesson needs it, and
  you don't need a GitHub account to follow the series: each lesson's
  starting point can be cloned without one. But a copy on GitHub is a
  backup, a way to share your work, and an everyday habit worth having.
  It takes three things, each covered by GitHub's own pages (theirs, so
  they may change):
  [create an account](https://docs.github.com/en/account-and-profile/how-tos/account-management/creating-an-account-on-github),
  [create a new repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
  with nothing in it (no README, license or `.gitignore`, since your
  project already has them), and
  [push this project to it](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
- **Why pin exactly.** The `==` pins are a choice for a tutorial: the
  text you're reading was tested with exactly these versions, and your
  code must match it. An application would usually allow a range
  (`>=2.56.0`) and let `uv.lock` fix the exact versions.
- **One lockfile, every system.** `uv.lock` covers macOS, Linux and
  Windows at once. A few packages are installed only on some systems:
  Windows gets two that macOS and Linux don't. Everyone gets the same version of
  every package they do install.
- **`uv run` adds, `uv sync` also removes.** `uv run` installs whatever
  the lockfile lists and the environment lacks, but leaves anything extra
  you installed by hand. `uv sync` makes `.venv` match the lockfile
  exactly, removing extras.
- **What the `src/` layout protects you from.** With your code under
  `src/`, Python can't pick it up from the project folder by accident:
  the tests and commands you run use the installed package, so they fail
  if the install is wrong. (Setting `PYTHONPATH` by hand would get around
  this.)
- **When the banner appears.** It's written to stderr, and only when
  stderr is a terminal or a coding agent is running the command:
  redirecting the output (`> out.txt`) still shows it; redirecting stderr
  (`2> err.txt`) leaves it out. It's also left out under pytest and in
  CI, and goes away once observability is set up.
- **Installing less.** If you need only one provider, `pydantic-ai-slim`
  with extras such as `openai` installs a fraction of the packages. The
  [installation page](https://pydantic.dev/docs/ai/overview/install/)
  lists the extras.
- **Fixing findings for you.** A finding Ruff can fix itself is marked
  `[*]`, and `uv run ruff check --fix` fixes those, such as an unused
  import. Read the finding first: in code you copied, an unused import
  usually means a line is missing, and `--fix` would delete the clue.
  With the Ruff extension, most editors can format and fix on save, once
  you switch that on in the editor's settings.
- **Checking without changing.** `uv run ruff format --check` reports
  which files it would reformat and changes nothing, for places where
  nothing should change, such as a check run on every push.
- **Your editor and `.venv`.** Editors such as VS Code, PyCharm and
  Spyder ask which Python interpreter a project uses. Choose the one in
  the project, `.venv/bin/python`: then the editor finds `pydantic_ai`
  for its hints, and uses the versions of Ruff, mypy and pytest this
  project pins. In PyCharm that's enough to run the tests from the
  editor; other editors may also need their Python or testing extension
  set up, which their own docs cover.
- **Your editor's files.** Editors write settings folders into your
  project: PyCharm makes `.idea/`, VS Code `.vscode/`. They belong to
  you, not to the project, so keep them out of its `.gitignore` and
  ignore them for every repository at once, in a personal ignore file:

  ```bash
  printf '.idea/\n.vscode/\n' >> ~/.gitignore_global
  git config --global core.excludesFile ~/.gitignore_global
  ```

  Once your editor has made its folder, `git check-ignore -v .idea` names
  `~/.gitignore_global` as the rule that matched. (Unlike step 5's check,
  it prints nothing before the folder exists: a pattern ending in `/`
  matches only folders.)
- **`uv` itself.** The [`uv` docs](https://docs.astral.sh/uv/) cover
  projects, lockfiles and Python versions in depth.

## Next

You have a project and an agent that answers, but you haven't yet seen
what an agent does. In the next lesson you build your first real
`Agent`, look at the messages it sent and received (the same history you
met in the orientation, now for real), and script its model yourself,
still with no API key.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
