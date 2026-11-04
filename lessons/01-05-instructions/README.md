# A word on system prompts and instructions

## Welcome

Welcome back. In the last lesson the shop agent's instructions came
from two places: fixed text, and a function that names the shop.

This lesson is a short note, with nothing to build. It's about **system
prompts and instructions**: two ways to tell the model what its job is.
Your agents use `instructions=`. In the Pydantic AI docs, and in other
people's code, you'll also meet `system_prompt=`. They're nearly the same
thing, and there's one place they differ.

By the end, you'll be able to:

1. **Say how a system prompt and instructions differ, and why this
   project uses instructions.**

About 5 minutes.

## Concepts

### Two ways to describe the model's job

Both `system_prompt=` and `instructions=` tell the model how to behave.
For example, either can tell an agent to answer questions about prices:

```python
Agent(system_prompt="You answer questions about prices. Answer briefly.")
Agent(instructions="You answer questions about prices. Answer briefly.")
```

In either case, Pydantic AI supplies guidance to the model along with
the conversation. Providers may package that guidance differently, but
the important distinction here isn't the provider's request format.
It's **what happens to the guidance when you continue a conversation**.

### What the history keeps

The difference is in Pydantic AI's own record of a run, its list of
messages: the history. The two are kept in different places:

| | Where Pydantic AI keeps it | What a later request is sent |
|---|---|---|
| **system prompt** | Inside the first request, so it's one of the messages | The copy in the history |
| **instructions** | Beside the messages, not in them | The running agent's own, fresh every request |

For the single-run examples you've built so far, this distinction has
little practical effect. It matters when a run
continues an earlier conversation by being handed that conversation's
messages, which Chapter 2 does. Then a system prompt is the one saved at
the start: change it in your code, and the conversation keeps the old
one. Instructions are always what the agent says now. That's also why
`name_the_shop`, from the last lesson, works: it's called before every
request, so its sentence can't go stale.

### Which to use

The Pydantic AI docs recommend instructions "for most use cases", and
this project uses only instructions. If you ever give an agent both, the
system prompt is sent first and the instructions after it. Don't let
them disagree: which one the model then follows is up to the model, not
Pydantic AI.

## What you can now do

**You can say how a system prompt and instructions differ, and why this
project uses instructions.**
Both forms guide the model. Pydantic AI handles their history
differently: a system prompt is kept with the messages; instructions
come from the running agent on each request.

## Next

Your agent still answers in plain text. In the next lesson you ask an
agent for a typed answer instead, a value your code can use, and give it
a rule that sends the model back when that value is wrong.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
