# A real model

## Welcome

Welcome back. Chapter 1 ran every agent on a stand-in model, inside your
own program. That's how you learned what each piece does.

Chapter 2 begins a different kind of project. Instead of making a separate
little example in each lesson, you'll build **one terminal chat** and
keep improving it through the rest of the tutorial. First it needs a
real model — and this lesson is about that: setting up access to one,
and seeing what changes when a model makes its own decisions.

By the end, you'll be able to:

1. **Get an OpenAI key and keep it out of your code and out of git.**
2. **Run an agent from Chapter 1 on a real model, unchanged.**
3. **Read what a real run cost**, and keep it under a limit you set.

Tested on `gpt-4o-mini`. About 15 minutes.

## Concepts

### An API key is your account's password

A model is a service you pay for. Different providers — Anthropic,
OpenAI and others — offer their own models. You sign up with one and
get an API key: a secret string that works like a password for your
account. Anyone who has it can spend your money.

Never put the key in source code or commit it to git. In these lessons,
you'll supply `OPENAI_API_KEY` through your shell environment. Pydantic AI
reads it when the program uses the OpenAI provider. An environment
variable helps keep the key out of the repository; it does not make the
secret impossible for other programs on your computer to read.

### The model is one setting

In Chapter 1 every agent ran on a stand-in model that Pydantic AI
provides: `model=TestModel()`, or a `FunctionModel`. A real model is
passed the same way, as a name: `model="openai:gpt-4o-mini"`. The part
before the colon is the provider; the part after is the model.

The agent doesn't change. Its instructions, tools and code are the same;
only what answers is different. To keep it that way, the model's name
lives in one place in your project, and every script reads it from
there. Changing the model is then one line.

The next few lessons use `gpt-4o-mini`, a relatively inexpensive
OpenAI model, so you'll need an OpenAI API key. Later on, you'll plug
in any provider you choose.

### A real run costs money

In Chapter 1 the usage was made up: `TestModel` invented its token
counts, and nothing was billed. A real model consumes billable tokens.
Pydantic AI reports usage in `result.usage`, including a cost estimate
when pricing data is available.

The exact charge depends on the model, the tokens used, and current
provider pricing. Even small charges add up if a loop keeps making
requests. Set a modest budget before you start, and check whether your
provider treats that budget as an alert or a hard limit.

## What you'll build

Your key in an environment variable, the model's name in one setting,
and a script that runs the price agent from the tools lesson on the real
model.

```text
pydantic-tutor/
├── scripts/
│   └── first_real_run.py    new: the price agent on a real model
└── src/
    └── pydantic_tutor/
        └── model.py         new: the model's name, in one place
```

And the steps, in order:

1. **Get a key**, set a spending limit, and set the key in your shell.
2. **Name the model** in one setting.
3. **Run the agent on the real model.**

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson.

> [!TIP]
> **Check:** from the project folder, run:
>
> ```bash
> uv run hello_agent.py
> ```
>
> The last line should be:
>
> ```text
> success (no tool calls)
> ```

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works,
> skip this. To refresh a broken project, delete its `pydantic-tutor` folder
> first: you lose your work and anything else in it, including local
> credentials. Then, from the folder that should hold the project:
>
> ```bash
> git clone --branch 01-08-limits/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Get a key

> **Goal:** an OpenAI key, a spending limit on your account, and the key
> in an environment variable.

OpenAI's site changes its pages from time to time, so the names below
may differ a little from what you see. At the time of writing:

1. Sign in, or sign up, at [platform.openai.com](https://platform.openai.com).
2. Under **Billing**, add a small amount of credit. A key doesn't work
   on an account with none.
3. Under **Limits**, set a monthly budget you're comfortable with. A few
   dollars is far more than this Chapter needs.
4. On the [API keys](https://platform.openai.com/api-keys) page, create
   a key, and copy it. You'll only see it once.

One key works for every OpenAI model: you choose the model in code, not
when you create the key.

Now supply the key as an environment variable in the terminal where
you'll run the project. Replace the placeholder with your real key. Don't
paste the real value into a commit, screenshot, or chat message. For a
longer-lived setup, you can use a secret manager or your shell's startup
configuration; remember that a plaintext startup file also stores the
secret on disk.

One more variable while you're here: Chapter 2's checks read each script's
own output, so Pydantic AI's startup banner is only noise — hide it:

```bash
export OPENAI_API_KEY=your-key-here
export PYDANTIC_AI_NO_BANNER=1
```

Keep using this terminal for the following commands.

> [!TIP]
> **Check:** confirm the key is set without printing any part of it:
>
> ```bash
> test -n "$OPENAI_API_KEY" && echo 'Key is set' || echo 'Key is missing'
> ```
>
> You should see:
>
> ```text
> Key is set
> ```
>
> If you see `Key is missing`, set the variable in this terminal before
> continuing.

**What it means**

The Python program can access the key without keeping it in a tracked
file. This reduces the risk of an accidental commit; you must still
protect the key and your account.

Only programs started from a shell see it. Your editor's terminal does;
an app started from the Dock or a desktop menu may not. Run the lessons'
commands from a terminal.

## Step 2: Name the model

> **Goal:** the real model's name, in one place.

The chat will need its model name from somewhere, so you'll put it in
a small module the chat can import later. That keeps the choice
separate from the code that asks questions: when you change providers
or models, you change one line instead of hunting through the program.
The file holds a single setting, `CHAT_MODEL`.

Create `src/pydantic_tutor/model.py`:

```python
"""The real model the chat runs on, in one place.

Every script that uses a real model
reads it from here, so changing the model is one line.
"""

# The model used by the chat throughout this chapter.
CHAT_MODEL = "openai:gpt-4o-mini"
```

> [!TIP]
> **Check:** print it:
>
> ```bash
> uv run python -c "from pydantic_tutor.model import CHAT_MODEL; print(CHAT_MODEL)"
> ```
>
> You should see:
>
> ```text
> openai:gpt-4o-mini
> ```

**What it means**

The check confirms the setting can be imported by another Python file.
That's the whole setting: a provider, `openai`, and a model,
`gpt-4o-mini`. Every script in this Chapter reads `CHAT_MODEL`, so if you
ever want another model, you change this one line.

## Step 3: Run the agent on the real model

> **Goal:** run the price agent on the real model, and read what it cost.

This is a bridge from Chapter 1 into the chat you'll soon build. The
`price_agent` and its tools are familiar; the new piece is
`model=CHAT_MODEL`. Look at the final four lines too: they let you
inspect what happened instead of taking the answer on trust.

Create `scripts/first_real_run.py`:

```python
"""Run the price agent on a real model, unchanged.

Run it with `OPENAI_API_KEY` set:
`uv run scripts/first_real_run.py`.
"""

from pydantic_tutor.model import CHAT_MODEL
from pydantic_tutor.prices import price_agent

result = price_agent.run_sync(user_prompt="How much is a coffee?", model=CHAT_MODEL)

for message in result.all_messages():
    print(message.kind, [part.part_kind for part in message.parts])
print("output:", result.output)
print("requests:", result.usage.requests)
print("cost in dollars:", result.usage.cost)
```

Run it. This is the first time the script contacts the provider, so it
uses your key:

```bash
uv run scripts/first_real_run.py
```

Before you look at the output, write down a guess: how many requests
will the run make? A real model chooses its own route through the
tools, so compare the kind of messages and the final answer, not every
word or number.

And here's what each message line is, when it prints:

- `request ['user-prompt']` — your question went to the model: "How
  much is a coffee?"
- `response ['tool-call']` — the model asked the agent to run the
  `price_of` tool.
- `request ['tool-return']` — the agent sent the tool's answer back to
  the model.
- `response ['text']` — the model answered in words: a coffee costs 700
  cents.

> [!TIP]
> **Check:** you should see something like this:
>
> ```text
> request ['user-prompt']
> response ['tool-call']
> request ['tool-return']
> response ['text']
> output: A coffee costs 700 cents, or $7.00.
> requests: 2
> cost in dollars: 0.00004845
> ```
>
> Your output will differ, and that's expected. What must match: at
> least one `tool-call` and its `tool-return`, a `text` answer that says
> coffee is 700 cents, and a cost of a small fraction of a cent.

**What it means**

A real model read your question and the tool definitions, decided to
call `price_of`, and answered from what it returned. That's the loop
from the tools lesson, with a model that decides for itself instead of
following a script.

Deciding for itself means it doesn't always decide the same way. In
another of our runs, it called a tool twice before answering, and made
three requests instead of two:

```text
request ['user-prompt']
response ['tool-call']
request ['tool-return']
response ['tool-call']
request ['tool-return']
response ['text']
output: A coffee costs 700 cents, or $7.00.
requests: 3
cost in dollars: 0.00007560
```

Same agent, same question, a different path to the same answer. That's
why a real model's output can't be checked line for line, and why Chapter 1
used stand-ins to show each piece exactly.

The reported cost helps you understand the run. Use your provider's
billing dashboard to confirm actual charges.

## What you can now do

- **The key stays out of the source:** you supplied it through the
  environment rather than hard-coding it. That reduces risk, but is not
  a guarantee of security.
- **The model is one setting**, `CHAT_MODEL`. The agent from Chapter 1 ran
  on a real model without a single change.
- **A real run costs money,** and `result.usage.cost` tells you how much.
  Check billing too; a budget is not always an enforced hard cap.

## Next

You now have the real model connection that the growing chat program
will use. Next you'll build its first version: a terminal loop that
accepts questions and prints replies.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
