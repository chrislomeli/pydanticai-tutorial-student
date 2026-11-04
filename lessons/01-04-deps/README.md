# Dependencies

## Welcome

Welcome back. In the last lesson you gave an agent tools over a price
list that sat in the tools' own file.

This lesson is about **dependencies**, or **deps**: the information a
question needs right now, such as which shop it's about, passed to the
agent as an object with each run.

By the end, you'll be able to:

1. **Pass the information a run needs as deps, and read it in a tool and
   in the instructions.**
2. **Say why an agent needs deps**, rather than a value fixed in its
   code.

About 15 minutes.

## Concepts

### What deps are

Deps are the information a run needs right now: which shop the question
is about, which user is asking, a connection to a database. You put it in
one object, usually a small dataclass, and pass it with the run. The next
run can pass a different one.

The name is Pydantic AI's: the run's **dependencies**. It has nothing to
do with the package dependencies in `pyproject.toml`.

There are three pieces:

- **The agent says what type they are:** `deps_type=Shop`. `Shop` is a
  small class of our own.
- **Each run passes a value:** `run_sync(..., deps=Shop(...))`.
- **Your code reads it as `ctx.deps`.** `ctx` is the run's context, a
  `RunContext`. The agent passes it to your code; the model never does.

Two kinds of code read the deps in this lesson:

- **A tool** takes `ctx: RunContext[Shop]` as its first parameter. `ctx`
  isn't in the tool definition, so the model can't see the shop or pick
  one.
- **An instructions function**, decorated with `@agent.instructions`,
  takes `ctx` and returns text. Before each request, the agent calls it
  and adds the text to the agent's fixed instructions. That's how the
  model learns which shop this run is for: from a sentence your code
  wrote, not from the deps.

The model sees what your code makes of the deps (the instructions text,
and whatever a tool returns), never the deps themselves.

### Why not put it in the code?

You build an agent once, when your program starts, and run it many
times. In a real program those runs are for different users, accounts or
shops, often at the same moment.

A value fixed in the agent's code, like last lesson's price list, is the
same for every run. Deps let the code that starts a run decide what the
run is about. The same agent, with the same tools, answers for whichever
shop it's handed.

The model isn't given the deps object to choose a different shop. Your
program selects the shop before the run begins, and the tool reads it
from `ctx.deps`. That helps keep responsibilities clear, but it isn't
a security boundary by itself. In a real application, your tools must
still check that the caller is authorized to access that shop's data.

## What you'll build

A shop agent whose shop is the run's dependency, and a script that runs
it for two shops, on a stand-in model we give you.

```text
pydantic-tutor/
├── scripts/
│   └── two_shops.py        new: one agent, two shops
└── src/
    └── pydantic_tutor/
        ├── given/
        │   └── shop_model.py   given: the stand-in model
        └── shop.py         new: Shop, the agent, its tool and its instructions
```

And the steps, in order:

1. **Write the shop agent**, with a tool and instructions that read the
   deps.
2. **Run it for two shops.**

Each step ends with a **Check**. If it doesn't show what it says, stop
there: the next step builds on it.

## Before you start

Carry on in your `pydantic-tutor` project from the last lesson. This lesson hands you one
more stand-in model. From the project folder, fetch it:

```bash
uv run https://raw.githubusercontent.com/chrislomeli/pydanticai-tutorial-student/main/starter/get_starter.py shop-model
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
> git clone --branch 01-03-function-tools/end --single-branch https://github.com/chrislomeli/pydanticai-tutorial-student.git pydantic-tutor
> cd pydantic-tutor
> uv sync
> ```
>
> The copy also has a `tests/` folder, our own checks for each lesson. You
> don't need it. Then run the check above.

## Step 1: Write the shop agent

> **Goal:** a tool and instructions that read the shop from the run's
> deps.

We'll move the shop's prices out of the agent's fixed code and into
**run-specific dependencies**. Look at `RunContext[Shop]` in the tool:
`ctx.deps` is how the tool receives the particular shop selected for
this run. The model doesn't select that object.

Create `src/pydantic_tutor/shop.py`:

```python
"""An agent that answers for whichever shop a run is about.

The shop is the run's dependency:
each run passes one, and the tool and the instructions read it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass(frozen=True)
class Shop:
    """What one run needs: the shop it answers for."""

    name: str
    prices_in_cents: Mapping[str, int]


def price_of(ctx: RunContext[Shop], *, product: str) -> int:
    """Look up the price of one product, in cents.

    Args:
        product: The product's name.
    """
    # ctx.deps is the Shop this run was given; the model never sees it.
    prices = ctx.deps.prices_in_cents
    if product not in prices:
        raise ModelRetry(message=f"{ctx.deps.name} doesn't sell {product!r}.")
    return prices[product]


# No model and no shop here: each run says which model to use, and which
# shop it's about.
shop_agent = Agent(
    name="shop",
    deps_type=Shop,
    instructions="You answer questions about prices. Answer briefly.",
    tools=[price_of],
)


@shop_agent.instructions
def name_the_shop(ctx: RunContext[Shop]) -> str:
    """Tell the model which shop this run is for."""
    return f"You answer for {ctx.deps.name}."
```

`ctx` comes first in `price_of`, before the `*`, because the agent
passes it by position. The model's argument, `product`, stays
keyword-only.

Now run the agent without passing a shop. Guess what happens before you
run it:

```bash
uv run python -c "from pydantic_ai.models.test import TestModel; from pydantic_tutor.shop import shop_agent; shop_agent.run_sync(user_prompt='How much is a coffee?', model=TestModel())"
```

> [!TIP]
> **Check:** it fails. The last lines of the error are:
>
> ```text
>     return f"You answer for {ctx.deps.name}."
>                              ^^^^^^^^^^^^^
> AttributeError: 'NoneType' object has no attribute 'name'
> ```

**What it means**

The run started, and `ctx.deps` was `None`: no run passed a shop. The
first code to read it was `name_the_shop`, before the first request, so
the model was never asked anything.

`deps_type=Shop` tells mypy what `ctx.deps` is, so mypy checks a `deps=`
you pass. Nothing checks that you pass one at all.

## Step 2: Run it for two shops

> **Goal:** run the same agent for two shops, and see that only the deps
> change.

The script runs the agent twice, once for each shop, with a given model,
`asks_for_coffee()`. On the first request it prints what it was sent:
the instructions, and the arguments of each tool. Then it asks for the price
of coffee, and answers with it.

Before you run it, guess: will the model see the shop's prices?

This script runs the **same agent twice**, each time with a different
`Shop` passed as `deps`. Watch the `deps=shop` argument in `run_sync`:
that's what changes between the runs, not the agent definition.

Create `scripts/two_shops.py`:

```python
"""Run one agent for two shops: only the run's dependencies change."""

from pydantic_tutor.given.shop_model import asks_for_coffee
from pydantic_tutor.shop import Shop, shop_agent

shops = [
    Shop(name="Corner Cafe", prices_in_cents={"coffee": 350, "tea": 300}),
    Shop(name="Station Kiosk", prices_in_cents={"coffee": 500}),
]
for shop in shops:
    print(shop.name)
    result = shop_agent.run_sync(
        user_prompt="How much is a coffee?", model=asks_for_coffee(), deps=shop
    )
    print("  output:", result.output)
```

```bash
uv run scripts/two_shops.py
```

> [!TIP]
> **Check:** after the banner:
>
> ```text
> Corner Cafe
>   instructions: 'You answer questions about prices. Answer briefly.\n\nYou answer for Corner Cafe.'
>   price_of's arguments: ['product']
>   output: Coffee is 350 cents.
> Station Kiosk
>   instructions: 'You answer questions about prices. Answer briefly.\n\nYou answer for Station Kiosk.'
>   price_of's arguments: ['product']
>   output: Coffee is 500 cents.
> ```

**What it means**

Same agent, same tool, same model, same question. Only `deps=` changed,
and each run got its own shop's price.

The model didn't see the prices. It saw two things:

- **The instructions**: the fixed sentence, a blank line (the `\n\n`),
  then the sentence `name_the_shop` wrote for this run.
- **The tool's arguments**: only `product`. `ctx` isn't there.

It learned a price only when the tool returned one: `350` in the first
run, `500` in the second.

## What you can now do

- **Deps are the information a run needs right now**, in one object you
  pass with the run, and it can be different on every run. The agent
  declares its type with `deps_type`, each run passes one with `deps=`,
  and your tools and instructions read it as `ctx.deps`. The model sees
  what your code passes on, never the object itself.
- **One agent serves many runs**, so a value fixed in its code is the
  same for all of them. Deps let each run carry its own, and because the
  model isn't given the deps object, it doesn't select the shop for
  that run. Tools still enforce access permissions.

## Next

Your agent's instructions now come from two places: fixed text, and a
function that reads the deps. In the Pydantic AI docs you'll also meet a
**system prompt**, which looks like a third way to say the same thing.
The next lesson is a short note on how the two compare.

---

*Tested with `pydantic-ai` 2.56.0 and `pydantic-ai-harness` 0.56.0.*
