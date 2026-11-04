# Limits, retries and timeouts

## Welcome

Welcome back. In the last lesson you saw an agent fail on a plain
"Hello", because its answer had no room for words.

This lesson asks one practical question: **what should your program do
when a run can't proceed normally?** A tool problem may be something
the model can recover from. A run that keeps calling tools needs a
stopping point. A slow tool needs a deadline. These are different kinds
of trouble, and each calls for a different safeguard.

By the end, you'll be able to:

1. **Tell the model whether a failed tool call can be fixed**, with
   `ModelRetry` or `ToolFailed`.
2. **Stop a runaway run** with a request limit.
3. **Give up on a slow tool** with a tool timeout.

About 15 minutes.

## Concepts

There are three different decisions here: **can the model recover from
this tool failure?**, **how many requests should the run be allowed to
make?**, and **how long should the agent wait for a tool?** A retry, a
request limit, and a timeout answer different questions. None replaces
the others.

### Two ways a tool can fail

You met `ModelRetry` in the tools lesson: the tool tells the model "fix
your call", and the model gets one more try. That fits a mistake the
model can fix, like a misspelt name.

Some failures can't be fixed by asking again. If a supplier's stock
feed is down, it's down on the tenth ask too. For that, the tool raises
**`ToolFailed`**: "this won't work". The model gets a tool return marked
**failed**, with your message, and the run goes on, so the model can
tell the user.

| The tool raises | The model gets | The run |
|---|---|---|
| `ModelRetry` | Your message, and "try again". One retry per tool, by default | Goes on; a second miss ends it |
| `ToolFailed` | A tool return marked failed, with your message | Goes on |
| Any other exception | Nothing | Ends, with the exception in your code |

### A limit on requests

Nothing in the tool loop makes a model decide it's done. A model that
keeps calling tools keeps the run going, and with a real model every
request costs.

So you give the run a limit: `usage_limits=UsageLimits(request_limit=5)`.
Before each request the agent checks the count, and a request that would
go over is never sent. The run raises `UsageLimitExceeded` in your code
instead. A run with no limit of your own stops at 50 requests.

That matters most for `ToolFailed`. A tool that fails the same way every
time uses no retries, so nothing but the request limit stops a model
that keeps asking.

### A limit on time

A tool can wait on something that never answers. `tool_timeout=`, on
the agent, gives up on a tool call after that many seconds, and the
model gets a retry prompt saying it timed out.

A similar setting, `ModelSettings(timeout=...)`, bounds a request to a
real model. The stand-in models here send no request anywhere, so you'll
meet it with a real one.

## What you'll build

A stock agent with a tool that can fail in two ways, plus three scripts
that show recovery, a stopping limit, and a timeout. We'll use stand-in
models so each failure happens predictably.

```text
pydantic-tutor/
├── scripts/
│   ├── stock_check.py        new: a misspelt product, then one whose feed is down
│   ├── runaway.py            new: a model that never stops, and a limit
│   └── slow_tool.py          new: a tool that takes too long
└── src/
    └── pydantic_tutor/
        ├── given/
        │   └── limit_models.py  given: the stand-in models
        └── stock.py          new: the stock agent and its tool
```

And the steps, in order:

1. **Tell the model a tool can't work**, and compare it with a retry.
2. **Stop a runaway**, with a request limit.
3. **Give up on a slow tool**, with a timeout.

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. This lesson hands you more
stand-in models. From the project folder, fetch them:

```bash
uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py limit-models
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
> git clone --branch 01-07-output-trap/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Tell the model a tool can't work

> **Goal:** a tool that raises `ModelRetry` for a mistake the model can
> fix, and `ToolFailed` for one it can't.

We'll give one tool **two different ways to report trouble**. A
misspelled product raises `ModelRetry`, asking the model to correct its
call. A supplier feed that's down raises `ToolFailed`, because another
attempt won't repair the feed. Look at those two branches inside
`stock_of` as you read the code.

Create `src/pydantic_tutor/stock.py`:

```python
"""An agent with one tool that can fail in two ways."""

from pydantic_ai import Agent, ModelRetry, ToolFailed

IN_STOCK = {"apple": 12, "bread": 8, "coffee": 40}
# Products whose stock feed is down, so their stock can't be read.
FEED_DOWN = frozenset({"bread"})


def stock_of(*, product: str) -> int:
    """Look up how many of one product are in stock.

    Args:
        product: The product's name: apple, bread or coffee.
    """
    if product not in IN_STOCK:
        # The model wrote `product`: tell it what to fix.
        raise ModelRetry(message=f"There is no product {product!r}.")
    if product in FEED_DOWN:
        # Asking again won't help: the feed is down. Say so, and the model
        # can tell the user instead of retrying.
        raise ToolFailed(message=f"The stock feed for {product} is down.")
    return IN_STOCK[product]


# No model here: each run says which model to use.
stock_agent = Agent(
    name="stock",
    instructions="You answer questions about stock. Answer briefly.",
    tools=[stock_of],
)
```

Next we'll trigger both branches with a stand-in model,
`misspells_then_asks_for_bread()`. It first asks for the misspelled
`"bred"`, then for `"bread"`. **Notice that the script itself doesn't
handle the two tool failures:** the agent carries their messages back
to the model, which then answers.

Create `scripts/stock_check.py`.

```python
"""Ask about two products the tool can't answer for: one misspelt, one whose feed is down."""

from pydantic_tutor.given.limit_models import misspells_then_asks_for_bread
from pydantic_tutor.stock import stock_agent

result = stock_agent.run_sync(
    user_prompt="How much bread is there?", model=misspells_then_asks_for_bread()
)
print("output:", result.output)
```

```bash
uv run scripts/stock_check.py
```

> [!TIP]
> **Check:** after the banner:
>
> ```text
> asked to retry: There is no product 'bred'.
>
> Fix the errors and try again.
> tool return (failed): {"error":"The stock feed for bread is down."}
> output: I can't check bread right now.
> ```

**What it means**

Two failures, two rows of the table in Concepts:

- `bred` got a retry prompt: your `ModelRetry` message, then "Fix the
  errors and try again."
- `bread` got a tool return marked `failed`, with your `ToolFailed`
  message.

The run went on after both, and the model told the user what it could.
The difference is what you tell the model: "fix it" for a misspelling,
"this won't work" for a feed that's down.

## Step 2: Stop a runaway

> **Goal:** run the stock agent on a model that never stops calling the
> tool, and stop it with a request limit.

The model is given, `keeps_asking_for_bread()`: on every request it asks
for bread, and prints a line. Every call gets `ToolFailed`, which uses no
retries.

The given model keeps requesting the same unavailable stock, so a
normal tool retry won't stop it. **Look at `UsageLimits(request_limit=5)`
in the call:** this puts a cap on the number of model requests in this
run, and the `except` block shows how your code learns the limit was hit.

Create `scripts/runaway.py`:

```python
"""Stop a run whose model never stops calling a tool, with a request limit."""

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from pydantic_tutor.given.limit_models import keeps_asking_for_bread
from pydantic_tutor.stock import stock_agent

try:
    stock_agent.run_sync(
        user_prompt="How much bread is there?",
        model=keeps_asking_for_bread(),
        # This job needs two or three requests; five is room to spare.
        usage_limits=UsageLimits(request_limit=5),
    )
except UsageLimitExceeded as error:
    print("stopped:", error)
```

Before you run it, guess: with `request_limit=5`, how many `request`
lines will you see?

```bash
uv run scripts/runaway.py
```

> [!TIP]
> **Check:** after the banner, five requests, then the stop:
>
> ```text
> request 1: the model asks for bread
> request 2: the model asks for bread
> request 3: the model asks for bread
> request 4: the model asks for bread
> request 5: the model asks for bread
> stopped: The next request would exceed the request_limit of 5. Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
> ```

**What it means**

Five, not six. The agent checked before each request, and the sixth
would have gone over, so it was never sent. Your `except` caught
`UsageLimitExceeded`.

Five failed calls, and no retry was used, because `ToolFailed` doesn't
use one. Without the limit, this model would have made 50 requests
before the run stopped. With a real model, that's 50 paid requests for
nothing.

## Step 3: Give up on a slow tool

> **Goal:** give an agent a tool timeout, and see a slow tool time out.

The script builds a small agent of its own, whose one tool takes two
seconds. The agent gives up after half a second. The model is given,
`calls_once_then_answers()`: it calls the tool, and answers with what
comes back.

Now we'll limit **how long the agent waits for a tool**, rather than how many requests it makes.
The `wait_for_delivery` tool sleeps for two seconds; `tool_timeout=0.5`
tells the agent to stop waiting sooner. Notice that the tool is
`async def`, so the waiting operation can be cancelled.

Create `scripts/slow_tool.py`:

```python
"""Give up on a tool that takes too long, with a tool timeout."""

import asyncio

from pydantic_ai import Agent

from pydantic_tutor.given.limit_models import calls_once_then_answers


async def wait_for_delivery() -> str:
    """Wait for the supplier to confirm a delivery."""
    await asyncio.sleep(delay=2)  # the supplier is slow to answer
    return "Delivery confirmed."


# Half a second, so the timeout shows quickly. A real tool gets seconds,
# set from what it waits on.
slow = Agent(name="deliveries", tools=[wait_for_delivery], tool_timeout=0.5)
result = slow.run_sync(
    user_prompt="Is the delivery confirmed?", model=calls_once_then_answers()
)
print("output:", result.output)
```

```bash
uv run scripts/slow_tool.py
```

> [!TIP]
> **Check:** after the banner, about half a second later:
>
> ```text
> asked to retry: Timed out after 0.5 seconds.
>
> Fix the errors and try again.
> output: The supplier didn't answer.
> ```

**What it means**

The tool never returned "Delivery confirmed." After half a second the
agent gave up on the call, and the model got a retry prompt saying so.
The timeout uses the tool's one retry, like any `ModelRetry`.

`wait_for_delivery` is `async def`, so it was cancelled, and the script
ends about half a second in. A plain `def` tool can't be cancelled: the
agent stops waiting, but the function runs on to its end. If a tool
might hang, make it `async`.

## What you can now do

- **Recover when recovery is possible.** Use `ModelRetry` when the
  model can fix its call; use `ToolFailed` to report a problem another
  attempt won't fix. The run can continue so the model can explain it.
- **A request limit stops a runaway.** `usage_limits=UsageLimits(request_limit=N)`
  on the run; without one, a run stops at 50 requests.
- **A tool timeout stops a slow tool.** `tool_timeout=` on the agent
  gives up on a call and tells the model.

## Next

That's Chapter 1: an agent, a model, the messages between them, tools,
deps, typed answers and limits. Chapter 2 builds a chat.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
