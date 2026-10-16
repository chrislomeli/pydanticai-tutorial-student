# pydanticai-tutorial-student

The lessons and code for **pydantic-tutor**, a tutorial series on
[Pydantic AI](https://pydantic.dev/docs/ai/). It is an independent tutorial,
not a project of the Pydantic team.

You build one project, one short lesson at a time, starting from an empty
folder: an agent that works over a workspace. Along the way you write the
infrastructure a team would build for itself (tools, capabilities, memory,
approval, persistence, a context budget, background jobs) and then apply it
to real work.

## Who it is for

Python developers with light experience of agentic programming: you have
called a model's API, and perhaps tried a framework, but have not built an
agent application.

## How this repo is laid out

- **`main` holds the lessons**, in `lessons/<lesson id>/`. Start with
  [`lessons/01-01-setup/outline.md`](lessons/01-01-setup/outline.md).
- **The code lives on branches**, two per lesson: `<lesson id>/start`, what
  the lesson begins from, and `<lesson id>/end`, what it leaves you with.
  Cloning `main` gives you the lessons, not the code.

The first lesson, setup, starts in an empty folder of your own, outside this
repo, and has you build the project with `uv`. From the second lesson on,
you carry on in that project. If you fall behind or want a clean start,
clone this repo, check out the lesson's start branch, and compare your work
with its end:

```bash
git checkout <lesson id>/start
# ... follow the lesson ...
git diff <lesson id>/start <lesson id>/end    # what the lesson adds
```

## What you need

- macOS or Linux. On Windows, use WSL.
- `git`, and [`uv`](https://docs.astral.sh/uv/) 0.13.1 or newer. `uv`
  installs Python for you; the setup lesson has install guides.
- An API key for one model provider, from the lesson that first talks to a
  live model. Until then, every lesson runs without one.

## Lessons

| Lesson | What you do |
|---|---|
| [01-01 Setting up the environment](lessons/01-01-setup/outline.md) | Create a pinned project with Pydantic AI and the harness, and run a first agent with no API key |

Written against `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0, on
Python 3.14.
