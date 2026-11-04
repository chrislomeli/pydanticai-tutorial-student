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

- **`main` holds the lessons**, in `lessons/<lesson id>/`, listed
  in order under [Lessons](#lessons) below.
- **The code lives on branches**, one per lesson: `<lesson id>/end`, what the
  lesson leaves you with.
  Cloning `main` gives you the lessons, not the code.

The first lesson, setup, starts in an empty folder of your own, outside this
repo, and has you build the project with `uv`. From the second lesson on,
you carry on in that project. If you fall behind or want a clean start,
clone this repo at the previous lesson's end branch, and compare your work
with this lesson's end:

```bash
git checkout <previous lesson id>/end
# ... follow the lesson ...
git diff <previous lesson id>/end <lesson id>/end    # what the lesson adds
```

## What you need

- macOS or Linux. On Windows, use WSL.
- `git`, and [`uv`](https://docs.astral.sh/uv/) 0.13.1 or newer. `uv`
  installs Python 3.14 for you; the setup lesson has install guides.
- An API key for one model provider, from the lesson that first talks to a
  live model. Until then, every lesson runs without one.

## Lessons

Each lesson's page, and the branches it starts and ends on. A lesson with
no branches is one to read.

### Chapter 0: Before you start

| Lesson | What you do | End |
|---|---|---|
| [00-01 From chat window to agent](lessons/00-01-orientation/README.md) | A first look at agentic programming: find the model, instructions, messages and tools in a chat app you already use, then read the same session as a Pydantic AI program. | — |
| [00-02 The tools: uv, git, Ruff and mypy](lessons/00-02-tools/README.md) | What uv, git, Ruff and mypy are for, and what this series asks of each, before the setup lesson asks you to run them. | — |
| [00-03 Pick your operating system](lessons/00-03-operating-systems/README.md) | Pick the install page for your operating system — macOS, Linux, or Windows via WSL 2 — before the setup lesson. | — |
| [00-04 Install the tools: macOS](lessons/00-04-macos-installs/README.md) | Install or upgrade git and uv on macOS, for the setup lesson. | — |
| [00-05 Install the tools: Linux](lessons/00-05-linux-installation/README.md) | Install or upgrade git and uv on Ubuntu, for the setup lesson. | — |
| [00-06 Install the tools: Windows](lessons/00-06-windows-installation/README.md) | Set up WSL 2 on Windows, check it works, connect VS Code to it, then install git and uv inside it, for the setup lesson. | — |
| [00-07 The example you'll work with](lessons/00-07-the-tinyjobs-example/README.md) | tinyjobs, the small project your agent works on for the whole series: what it does, what your agent will be asked about it, and the codebase API you call to read it. | — |

### Chapter 1: The agent

| Lesson | What you do | End |
|---|---|---|
| [01-01 Setting up the environment](lessons/01-01-setup/README.md) | Create a pinned Python project with Pydantic AI and the harness installed, check its code with Ruff and mypy, and run a first agent with no API key. | [`01-01-setup/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-01-setup/end) |
| [01-02 An agent, and a model you can script](lessons/01-02-agent/README.md) | What a model is, what an agent is, and the messages they pass; read a run's messages, and write a model of your own as a Python function. | [`01-02-agent/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-02-agent/end) |
| [01-03 Tools, and the loop that calls them](lessons/01-03-function-tools/README.md) | What a tool is, what the model is told about it, the loop a tool call makes, and why a tool checks the arguments the model wrote. | [`01-03-function-tools/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-03-function-tools/end) |
| [01-04 Dependencies](lessons/01-04-deps/README.md) | Pass the information a question needs, as an object, with each run of an agent; read it in a tool and in the instructions; and see why the model never gets the object itself. | [`01-04-deps/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-04-deps/end) |
| [01-05 A word on system prompts and instructions](lessons/01-05-instructions/README.md) | Why Pydantic AI offers system prompts and instructions, and what changes when conversation history is reused. | [`01-05-instructions/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-05-instructions/end) |
| [01-06 Structured output: a typed answer](lessons/01-06-typed-answer/README.md) | Explain the division of work between the model, Pydantic AI and Pydantic, then show what validation can and cannot establish with an Invoice. | [`01-06-typed-answer/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-06-typed-answer/end) |
| [01-07 The structured-output trap](lessons/01-07-output-trap/README.md) | See why an agent's output types must fit its job: an Invoice-only extractor fails on a greeting, while an assistant can permit text and invoices. | [`01-07-output-trap/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-07-output-trap/end) |
| [01-08 Limits, retries and timeouts](lessons/01-08-limits/README.md) | Tell the model a tool call can be fixed or can't, stop a run that never ends with a request limit, and give up on a tool that takes too long. | [`01-08-limits/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/01-08-limits/end) |

### Chapter 2: A chat

| Lesson | What you do | End |
|---|---|---|
| [02-01 A real model](lessons/02-01-real-model/README.md) | Get an OpenAI key, keep it out of git, name the model in one place, and run a Chapter 1 agent on a real model, cost included. | [`02-01-real-model/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/02-01-real-model/end) |
| [02-02 A chat in the terminal](lessons/02-02-console-chat/README.md) | Write a chat in the terminal, a loop around the agent's runs, and watch it forget what you said a moment ago. | [`02-02-console-chat/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/02-02-console-chat/end) |
| [02-03 A chat that remembers](lessons/02-03-history/README.md) | Hand each run the messages of the chat so far, so the chat remembers, and watch what the model is sent grow. | [`02-03-history/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/02-03-history/end) |
| [02-04 Streaming the answer](lessons/02-04-streaming/README.md) | Stream the chat's answers as the model writes them, and see that the run, history included, is the same as before. | [`02-04-streaming/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/02-04-streaming/end) |
| [02-05 Web search, a capability](lessons/02-05-web-search/README.md) | Give the chat web search with one prebuilt capability, and see it answer what the model can't know. | [`02-05-web-search/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/02-05-web-search/end) |

### Chapter 3: The application and a workspace

[Chapter overview](lessons/03-00-application-and-workspace/README.md)

| Lesson | What you do | End |
|---|---|---|
| [03-01 Settings and a composition root](lessons/03-01-settings/README.md) | Gather what the chat is configured with into one typed Settings, build its objects in one function, and make the app fail at startup, by name. | [`03-01-settings/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/03-01-settings/end) |
| [03-02 Models and providers](lessons/03-02-models/README.md) | Give the existing chat an explicit, secret-safe way to construct OpenAI and Anthropic models. | [`03-02-models/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/03-02-models/end) |
| [03-03 A workspace](lessons/03-03-workspace/README.md) | Attach a LocalWorkspace to the chat's agent at the composition root, and show with a keyless script that it is a starting folder, not a fence. | [`03-03-workspace/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/03-03-workspace/end) |
| [03-04 Let the chat read your project](lessons/03-04-files/README.md) | Give the continuing chat file tools, then let a real model discover, read and investigate the tinyjobs project. | [`03-04-files/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/03-04-files/end) |
| [03-05 Let the chat run a command](lessons/03-05-shell/README.md) | Give the continuing chat a limited shell, then use a real model to run the tinyjobs tests. | [`03-05-shell/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/03-05-shell/end) |

### Chapter 4: Capabilities

[Chapter overview](lessons/04-00-capabilities/README.md)

| Lesson | What you do | End |
|---|---|---|
| [04-01 Toolsets](lessons/04-01-toolsets/README.md) | Package supplied workspace tools in a FunctionToolset, attach it to the chat, and try prefixed and filtered views. | [`04-01-toolsets/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-01-toolsets/end) |
| [04-02 Agent runs with hooks](lessons/04-02-hooks/README.md) | Write three hooks that record each request and tool call of a run in a log file, and read the agent's loop from what they write. | [`04-02-hooks/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-02-hooks/end) |
| [04-03 Put the activity log in the chat](lessons/04-03-activity-log/README.md) | Connect the activity log to the chat through the composition root, choose what it records from Settings, and read the log of a real conversation. | [`04-03-activity-log/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-03-activity-log/end) |
| [04-04 Change tool calls with hooks](lessons/04-04-guards/README.md) | Write hooks that refuse a chained shell command and remove this project's tokens from tool results, and see why their place in the list decides what the activity log records. | [`04-04-guards/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-04-guards/end) |
| [04-05 Build a custom capability](lessons/04-05-capability/README.md) | See what each capability in the chat's list brings, then turn the token-removing hook into a capability of your own that also tells the model what it did. | [`04-05-capability/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-05-capability/end) |
| [04-06 Move the chat prompt into a file](lessons/04-06-prompt/README.md) | Move the chat's prompt out of Python into a file that a capability of your own reads, without changing what the model is sent, and predict the order in which instructions from several places reach the model. | [`04-06-prompt/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-06-prompt/end) |
| [04-07 Save notes between chats](lessons/04-07-memory/README.md) | Give the chat a notebook of files it reads and writes across sessions, with a list tool of your own, and see what the library's memory injection would add to every request and why the chat turns it off. | [`04-07-memory/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-07-memory/end) |
| [04-08 Separate agent settings from capabilities](lessons/04-08-seam/README.md) | Sort what the chat agent is from what it does: the constructor keeps its name, output type and request defaults, a list of capabilities holds the rest, and the capabilities whose place matters declare it, so the list can be rearranged and added to safely. | [`04-08-seam/end`](https://github.com/chrislomeli/pydanticai-tutorial-student/tree/04-08-seam/end) |

Written against `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0, on
Python 3.14.
