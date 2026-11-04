# The structured-output trap

## Welcome

Welcome back. In the last lesson the extractor learned to answer with an
`Invoice`, and to catch its own errors.

This lesson is about a **trap** in structured output: what happens when
an agent that can only answer with an `Invoice` is asked a question that the agent can't answer with an `Invoice`?

For example, what if we just say something like "Hello"?

By the end of this lesson, you'll be able to:

1. **Explain why an `Invoice`-only extractor fails on a greeting**, even when the model tries to answer.
2. **Allow text or an `Invoice`** when an agent's job requires both.

About 10 minutes.

## Concepts

### The output type must fit the job

The extractor has `output_type=Invoice`, so a run can end one way only:
with an `Invoice`. Words are not an answer.

That's right for the extractor. Your code calls it with a document, and
wants nothing back but the record.

It isn't right for every job. An assistant that greets people and also
extracts invoices must be able to return words when there's no invoice.
If its only permitted output is `Invoice`, a greeting isn't an acceptable
answer: the agent asks the model to try again, then the run fails. A
model forced to return an `Invoice` might instead invent one.

The lesson isn't that structured output and conversation can't mix.
It's that the **permitted output types must match the agent's job**.

## What you'll build

One script, `scripts/output_trap.py`, that first says "Hello" to the
extractor, then tries an agent that permits either words or an `Invoice`.
Both use a stand-in model we give you.

```text
pydantic-tutor/
├── scripts/
│   └── output_trap.py          new: greet the extractor, then allow both output types
└── src/
    └── pydantic_tutor/
        └── given/
            └── greeting_model.py  given: the stand-in model
```

There are two steps: **see why the extractor fails on a greeting**,
then **allow an assistant to return words or an invoice**.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. This lesson hands you one
more stand-in model. From the project folder, fetch it:

```bash
uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py greeting-model
```

> [!TIP]
> **Check:** from the project folder, run:
>
> ```bash
> uv run hello_agent.py
> ```
>
> After the banner, the last line should be:
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
> git clone --branch 01-06-typed-answer/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Say hello to the extractor

> **Goal:** greet the agent whose answer must be an `Invoice`, and read
> how the run fails.

The model in this lesson is given, `greets_on_hello()`. Told "Hello", it
answers in words, as a real model would. It prints whether the agent
offered it a way to answer in words, and anything it's sent back.

First we'll deliberately give the extractor a job it cannot finish:
answering a greeting when its only permitted output is an `Invoice`.
**Watch the `try`/`except`:** it lets us inspect the failure instead of
ending the script at the exception.

Create `scripts/output_trap.py`:

```python
"""Say "Hello" to an agent whose answer must be an Invoice, then to one that may answer in words."""

from pydantic_ai.exceptions import UnexpectedModelBehavior

from pydantic_tutor.given.greeting_model import greets_on_hello
from pydantic_tutor.invoices import extractor

print('The extractor, told "Hello":')
try:
    extractor.run_sync(user_prompt="Hello", model=greets_on_hello())
except UnexpectedModelBehavior as error:
    print("  the run failed:", error)
```

Before you run it, guess: the agent will send the model back once. What
will the message say?

```bash
uv run scripts/output_trap.py
```

> [!TIP]
> **Check:** after the banner, you should see:
>
> ````text
> The extractor, told "Hello":
>   a text answer offered: False
>   a text answer offered: False
>   sent back to the model:
> 1 validation error:
> ```json
> [
>   {
>     "type": "json_invalid",
>     "loc": [],
>     "msg": "Invalid JSON: expected value at line 1 column 1"
>   }
> ]
> ```
>
> Fix the errors and try again.
>   the run failed: Exceeded maximum output retries (1)
> ````

**What it means**

`a text answer offered: False`, on both requests: the agent gave the
model no way to answer in words. The model greeted anyway.

The agent could only read the greeting as an `Invoice`, written as JSON.
"Hello! Send me an invoice…" isn't JSON, so it sent back `Invalid JSON`.
The model greeted again, the one retry was spent, and the run ended in
your `except` block.

That message is the trap's disguise. It says nothing about a greeting or
an output type. Find it in a log, and you'd suspect a broken request.
The real cause is that the agent's output type has no room for words.

## Step 2: Allow words or an Invoice

Now imagine an assistant with two responsibilities: greeting people and
extracting invoices when they paste one. Its permitted outputs should
reflect both jobs. Set `output_type=[str, Invoice]`. A run can then
return a `str` or an `Invoice`, and your code checks which it received
with `isinstance`.

This second agent keeps the invoice option but also permits an ordinary
string. **Notice both changes:** `output_type=[str, Invoice]` accepts
one of two result types, and `isinstance(answer, Invoice)` lets your
Python code handle each kind appropriately.

In `scripts/output_trap.py`, replace the imports at the top with these.
`Agent`, `DOCUMENT` and `Invoice` are new:

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from pydantic_tutor.given.greeting_model import greets_on_hello
from pydantic_tutor.given.invoice_model import DOCUMENT
from pydantic_tutor.invoices import Invoice, extractor
```

Then add this code to the end of the file:

```python
# Words or an Invoice: either ends a run. With a list, the type parameters
# say so to the type checker: the deps type (none), then the output type.
desk = Agent[None, str | Invoice](
    name="invoice_desk",
    output_type=[str, Invoice],
    instructions="Greet people briefly. When you are given an invoice, extract it.",
)
for prompt in ["Hello", DOCUMENT]:
    answer = desk.run_sync(user_prompt=prompt, model=greets_on_hello()).output
    if isinstance(answer, Invoice):
        print("an Invoice:", answer.number, answer.total)
    else:
        print("words:", answer)
```

Run it again, on the same given model. After the extractor's lines, it prints:

```text
  a text answer offered: True
words: Hello! Send me an invoice and I will read it.
  a text answer offered: True
an Invoice: INV-0042 85.50
```

**What it means**

The agent can now accept a greeting as text while still returning an
`Invoice` for a document. Structured output hasn't disappeared. We've
chosen permitted outputs that fit both responsibilities.

## What you can now do

- **The output type must fit the job.** An `Invoice`-only extractor
  cannot accept a greeting as its final answer. An assistant that needs
  words and invoice records can use `output_type=[str, Invoice]`, and
  the calling code checks which kind of result it received.

## Next

The extractor gives up after one retry, and only a default of 50
requests stops an agent that keeps calling tools. The last lesson of
this chapter sets limits of your own: on requests, on retries, and on
time.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
