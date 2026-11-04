# The example you'll work with

*A short reading page, before the setup lesson. Nothing to install or
run yet.*

## Welcome

Before you build, a quick look at the two pieces of code this series
hands you. You don't need to learn either one — they're given, not
taught — but it helps to know what they're for.

The project you're building is a **coding assistant**: an agent that
answers questions about code. That makes the cast simple:

- **tinyjobs** is the sample codebase it practises on — a small
  job-queue program. What it *does* barely matters, and you'll never
  need to run it. It's just some code to answer questions about.
- **The codebase API** is a small ready-made module your agent's tools call
  to read a codebase — listing its modules, reading what each says
  about itself. It's ours, not part of Pydantic AI, and you'll see it
  again on this page.

That's the whole cast.

This is a reading page; no commands to run.

## What tinyjobs is, at a glance

As a program, tinyjobs is a job queue: you submit jobs, it runs them,
it remembers them. That's all you need to know about what it does —
what matters is what it *is*: about 250 lines of plain Python in five
modules, with a test suite.

One of its tests fails, and that's on purpose: tinyjobs has a known bug.
You don't need to find it. Later in the series, your agent does.

## What your agent will be asked

Over the series, your agent gets more ways to reach into tinyjobs, and
each one lets it answer a harder question. Roughly in order:

| Your agent can…                                       | So you can ask it…                           | From   |
|-------------------------------------------------------|----------------------------------------------|--------|
| List its modules and read what each says about itself | "What does the jobs module do?"              | Chapter 1 |
| Read its files, but never its secrets                 | "Explain the command's flags."               | Chapter 3 |
| Run its tests                                         | "The test suite has a failure. Why?"         | Chapter 3 |
| Remember what you decided                             | "What did we decide about logging?"          | Chapter 4 |
| Change it, once you approve                           | "Add a `--verbose` flag, and a test for it." | Later  |

You don't need to remember this table. Each lesson says what your agent
can do by the end of it.

## How it works, at a high level

tinyjobs is about 250 lines of plain Python, with no dependencies. Its
code is in five modules, under `src/tinyjobs/`:

| Module             | What it holds                                          |
|--------------------|--------------------------------------------------------|
| `tinyjobs`         | The package itself: what it offers to other code       |
| `tinyjobs.jobs`    | A job, and the queue that holds jobs                   |
| `tinyjobs.storage` | Saving the queue to its JSON file, and loading it back |
| `tinyjobs.cli`     | The `tinyjobs` command line                            |
| `tinyjobs.log`     | Where log messages go                                  |

Its tests are in `tests/`, beside `src/`. A real project also keeps
secrets out of its code, in a `.env` file. Later lessons give tinyjobs
one, with a fake secret in it, to check that your agent never shows it.

## Your code, and the example's

Your project holds three kinds of code, and it helps to keep them apart:

| Kind                 | Where                                                     | What you do with it                                                                                    |
|----------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Your code**        | `src/pydantic_tutor/` (all but `codebase.py`), `scripts/` | You write it, lesson by lesson. This is what the series teaches                                        |
| **The codebase API** | `src/pydantic_tutor/codebase.py`                          | Given to you. Your tools call it; you don't need to read it                                            |
| **tinyjobs**         | `sample_code/tinyjobs/`                                   | Given to you. Your agent reads it, and later changes it once you approve; you never change it yourself |

The **codebase API** is a small module that reads a codebase for your
agent's tools. It's ours, not part of Pydantic AI. You call two
functions:

- `codebase.list_modules(root=...)` gives a codebase's modules by name,
  such as `tinyjobs.jobs`.
- `codebase.read_docstring(root=..., module=...)` gives what one module says
  about itself, in its docstring, without running it.

It refuses a name the codebase doesn't have, so a name that sneaks
outside the codebase, such as `"../../.env"`, reads nothing. Later, when
your agent gets a workspace, the API reads through that instead, and the
lesson that does it says how.

Why hand you this code rather than have you write it? Because finding a
codebase's modules is ordinary Python. It isn't what this series
teaches. The tools that call it are, and you write those.

## What you can now do

**You can say what tinyjobs does, and what your agent will be asked
about it.** It's a small job queue with a command line, a test suite with
one known failure, and a plan to have your agent read it, run its tests
and, in the end, change it.

**You can tell your code from the example's.** You write the agent and
its tools. You're given the codebase API and tinyjobs. Your agent reads
tinyjobs; you never change it yourself.

## Next

Next, you set up the project for real: you install Pydantic AI with uv,
run your first agent with no API key, and check and commit your code.
tinyjobs joins the project when your agent gets its first tools, in the
function tools lesson.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
