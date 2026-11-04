# Web search, a capability

## Welcome

Welcome back. In the last lesson your chat learned to stream its
answers.

The chat can now carry on a conversation and stream its replies. But
what if the user asks for something that changed yesterday? In this
lesson you'll extend the **same chat agent** with web search, allowing
it to look up information that isn't reliably available from its
training alone.

By the end, you'll be able to:

1. **Give an agent web search** with one prebuilt capability.
2. **Say when the model searches, and what a search costs.**

Tested on `gpt-4o-mini`. About 10 minutes.

## Concepts

### A capability adds a feature

Without an external source, a model may not know recent events or the
latest version of a library. It can still sound confident, which is why
current facts should be checked against sources.

Giving a model a way to check usually means plumbing it yourself:
pick a search service, write the calls, parse what comes back, feed
the result to the model. Pydantic AI ships ready-made features that do
that plumbing for you, called **capabilities**. You add one in a line,
`capabilities=[...]`, and the agent gains what it does. `WebSearch()`
gives the model a way to search the web.

Later Chapters build capabilities of your own; this one you just use.

### The model decides when to search

Making web search available lets the model choose to request it. It
*may* search for the latest release of a library and *may* skip search
for a simple arithmetic question. Availability doesn't guarantee that
it will always make the best choice.

Here the searching itself is provided by OpenAI: when the model asks
to search, the provider's own search runs and its results come back in
the model's context — no search function of yours is involved. It's the
zero-plumbing option, and less flexible than one you'd write yourself;
`WebSearch` offers controls such as `allowed_domains` and `max_uses`,
but the searching happens on the provider's side. For a demo like this
one, that trade is worth it.

Provider support, pricing and billing rules can change; consult the
current provider pricing before relying on any per-search estimate.
Searching can add to the cost of your chat.

## What you'll build

One change to `chat/agent.py` gives your existing application another
capability. You don't need to rewrite its input loop, history handling
or streaming code.

```text
pydantic-tutor/
└── src/
    └── pydantic_tutor/
        └── chat/
            └── agent.py     changed: the agent can search the web
```

And the steps, in order:

1. **Ask the chat something it can't know.**
2. **Give it web search, and ask again.**

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. If you're in a new terminal, set `OPENAI_API_KEY` and `PYDANTIC_AI_NO_BANNER` again: an
environment variable lasts only as long as the terminal you set it in.

> [!TIP]
> **Check:** from the project folder, start the chat, send one line, and
> end it with an empty line:
>
> ```bash
> uv run python -m pydantic_tutor
> ```
>
> The agent should answer your line. 

> [!WARNING]
> **Joining here, or has your project gone wrong?** If your project works,
> skip this. To refresh a broken project, delete its `pydantic-tutor` folder
> first: you lose your work and anything else in it, including local
> credentials. Then, from the folder that should hold the project:
>
> ```bash
> git clone --branch 02-04-streaming/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Ask something it can't know

> **Goal:** see what the chat says about something newer than the model.

Start the chat, and ask:

```bash
uv run python -m pydantic_tutor
```

```text
What is the very latest release of pydantic-ai on PyPI, and when was it released?  Don't guess!
```

> [!TIP]
> **Check:** without search, the model may say it cannot confirm the
> latest version, or it may offer an unverified guess. Ours said:
>
> ```text
> agent> You can check the latest release of `pydantic-ai` on PyPI directly at [pypi.org](https://pypi.org/project/pydantic-ai/) for up-to-date information. My knowledge is not current beyond October 2021, so I can't provide the most recent release details.
>   (the model was sent 40 tokens)
> ```
>
> The wording can differ. What matters is that an answer without a current
> source isn't reliable evidence of the latest release.

**What it means**

The chat had no retrieval capability. It could provide background
knowledge, but it couldn't verify the current PyPI release. Even a
confident version number would need an up-to-date source.

## Step 2: Give it web search

> **Goal:** a chat that searches the web when it needs to.

Our chat already knows how to receive messages, preserve context, and
show replies as they arrive. The missing piece is a way to check
information that changes after a model was trained. Instead of writing
our own search function, we'll give the existing agent access to a
provider-backed search capability. The model can then request a search
as part of answering a question; making it available does not guarantee
that it will search every time.

This time the change isn't in the chat loop at all — `chat/loop.py`
stays exactly as it is. Open `src/pydantic_tutor/chat/agent.py` and
make two additions.

Add the import at the top of the file:

```python
from pydantic_ai.capabilities import WebSearch
```

Add the capability to the agent — one new parameter that we pass to the `Agent()`
call:

```python
capabilities=[WebSearch()]
```

With those edits in, `chat/agent.py` should look like this (the
docstring is updated to match):

```python
"""The agent the chat talks to.

No model here, as for every agent in this project: the chat passes
`CHAT_MODEL` on each run.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

chat_agent = Agent(
    name="chat",
    instructions="You are a helpful assistant. Answer briefly.",
    # The model may search the web when it needs to. Each search is billed
    # by the provider, on top of the tokens.
    capabilities=[WebSearch()],
)
```

Now compare two kinds of question in the same application. One asks for
a changing fact and benefits from a current source; the other can be
answered without retrieval. The contrast may help you observe when the
model chooses the new capability, though its decisions and token usage
can vary.

Start the chat again, and send the same question, then a simple one:

```text
What is the very latest release of pydantic-ai on PyPI, and when was it released?  Don't guess!
```

```text
What is 2 + 2?
```

Notice the contrast between these prompts: one requires current
information, while the other can be answered without looking anything
up. See how the model behaves, but don't assume it must search on one
and skip the other.

> [!TIP]
> **Check:** something like this:
>
> ```text
> you> What is the very latest release of pydantic-ai on PyPI, and when was it released?  Don't guess!
> agent> The latest release of pydantic-ai on PyPI is version 2.62.0, published on October 27, 2026. ([pypi.org](https://pypi.org/project/pydantic-ai/2.62.0/?utm_source=openai))
>   (the model was sent 8174 tokens)
> you> What is 2 + 2?
> agent> 2 + 2 equals 4.
>   (the model was sent 410 tokens)
> ```
>
> The version, date, source formatting and token counts will vary.
> Check the cited PyPI release yourself. A source-backed answer is the
> goal; a particular token count or search decision is not guaranteed.

**What it means**

The chat didn't need a new input loop or a different history mechanism
to look something up. Inside the run, the model decided it needed a
current source and asked to search; the provider's search ran, and its
results were added to what the model was sent — most of why the
input-token count jumped. The model then answered from what the search
brought back, and cited where it came from.

Here the fetched material *was* the answer: one fact, repeated back.
The same mechanism matters more when the finding feeds a bigger
question — a model that can reach for sources mid-run can weigh what
it finds instead of just repeating a single result.

Worth noticing who set the terms of that reach. The model could only
search because a capability said it may, on the provider's side. When you read that an agent "went rogue," look
for the tool boundary: something executed the action, and someone set
its scope. In a later Chapter you'll build the other half of that
boundary: an agent that asks before it acts.

The follow-up arithmetic question used far fewer tokens, which is the
cost contrast to notice; the search decisions themselves can vary from
run to run.

That release is newer than the one this series is written against,
2.56.0. The series stays on 2.56; the chat illustrated how to check a newer
release without changing the project dependencies.

## What you can now do

- **A capability adds a feature in one line:** `capabilities=[WebSearch()]`
  gave the chat web search.
- **The model can choose to search** when it has access to the
  capability. Verify important sources, and consult current provider
  pricing for search charges.

## Next

That's Chapter 2: you now have **one growing chat program**, connected to
a real model, with conversation history, streaming and optional web
search. Chapter 3 continues building this application, adding settings and
a workspace the agent can work in.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
