"""Checks that the project from the lesson on what a request costs is in place.

Our checks for the context-gauge lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done. None of them needs an API key or makes a
request: every run is on a scripted model, and every line "typed" at the
keyboard is scripted too.

None of them touches your own `work/` folder: each uses a folder of its own.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from pydantic_ai import ModelRequestContext, RunContext, models
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

# This file is tests/06_01_cost/test_cost.py: two folders below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TINYJOBS_ROOT: Path = PROJECT_ROOT / "sample_code" / "tinyjobs"
PREFIX = "PYDANTIC_TUTOR_"
SESSION = "scripts/cost_session.py"

# What the stand-in in these tests says each request cost, as a provider
# reports it: far more than the history's text, as a real request is.
REPORTED_INPUT = 5_000
REPORTED_OUTPUT = 10


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # None of this project's variables and no key, whatever the shell
    # running pytest has, and no real model call.
    for name in list(os.environ):
        if name.upper().startswith(PREFIX) or name.upper() == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


# The banner goes to stderr; turning it off keeps a failure readable.
CHILD_ENV = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYTHONUNBUFFERED": "1"}


def run_script(*, name: str, cwd: Path, stdin: str = "") -> str:
    """Run one of the project's scripts from `cwd`, `stdin` piped in; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=CHILD_ENV,
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A stand-in project folder: the sample codebase and the lesson's scripts, nothing else."""
    root = tmp_path / "project"
    shutil.copytree(src=TINYJOBS_ROOT, dst=root / "sample_code" / "tinyjobs")
    (root / "scripts").mkdir()
    for script in ("fresh_copy.py", "cost_session.py"):
        # Before the lesson there is no cost_session.py: the test that runs
        # it fails there, not the fixture.
        if (PROJECT_ROOT / "scripts" / script).is_file():
            shutil.copy(src=PROJECT_ROOT / "scripts" / script, dst=root / "scripts")
    run_script(name="scripts/fresh_copy.py", cwd=root)
    return root


def list_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A stand-in that lists the modules, then answers; it reports usage as a provider does."""
    asked = sum(isinstance(message, ModelResponse) for message in messages)
    if asked == 0:
        parts: list[ToolCallPart | TextPart] = [
            ToolCallPart(tool_name="list_modules", args={})
        ]
    else:
        parts = [TextPart(content="Done.")]
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(input_tokens=REPORTED_INPUT, output_tokens=REPORTED_OUTPUT),
    )


@dataclass
class KeepLastMessage(AbstractCapability[object]):
    """Send the model only the last message of each request."""

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return replace(request_context, messages=request_context.messages[-1:])


@dataclass
class Keyboard:
    """Stands in for `input`: hands out `lines`, then ends input."""

    lines: list[str]
    _lines: Iterator[str] = field(init=False)

    def __post_init__(self) -> None:
        self._lines = iter(self.lines)

    def __call__(self, prompt: str = "") -> str:
        print(prompt, end="")
        line = next(self._lines, None)
        if line is None:
            raise EOFError
        return line


def test_the_gauge_reads_each_request_and_counts_what_the_provider_reported() -> None:
    # The lesson's point: one reading per request, and from the second on,
    # the reading starts from the tokens the last response reported (which
    # count the instructions and every tool definition), not from the
    # history's text alone. The first request has no such count: its
    # reading is far smaller. On a model with no stream function: the
    # gauge doesn't make the run stream.
    from pydantic_tutor import context_gauge
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    gauge = context_gauge.ContextGauge()
    agent.run_sync(
        user_prompt="What modules are there?",
        model=FunctionModel(function=list_then_answer),
        deps=CodebaseDeps(root=TINYJOBS_ROOT),
        capabilities=[gauge],
    )
    assert len(gauge.readings) == 2
    first, second = gauge.readings
    assert first.used_tokens < 1_000
    reported = REPORTED_INPUT + REPORTED_OUTPUT
    # Plus the list_modules return, estimated from its text.
    assert reported < second.used_tokens < reported + 200
    assert (second.window_tokens, second.resolved) == (200_000, False)


def test_the_gauge_reads_the_request_as_the_model_is_sent_it() -> None:
    # Innermost: listed first, it still reads each request after a
    # capability listed after it has cut the history down.
    from pydantic_tutor import context_gauge
    from pydantic_tutor.agent import agent
    from pydantic_tutor.codebase_tools import CodebaseDeps

    last: list[int] = []
    for extra in ([], [KeepLastMessage()]):
        # Counted here, not from the messages: the cut hides them.
        asked: list[int] = []

        def list_twice(
            messages: list[ModelMessage], info: AgentInfo, asked: list[int] = asked
        ) -> ModelResponse:
            asked.append(1)
            usage = RequestUsage(input_tokens=REPORTED_INPUT, output_tokens=10)
            if len(asked) <= 2:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_modules", args={})],
                    usage=usage,
                )
            return ModelResponse(parts=[TextPart(content="Done.")], usage=usage)

        gauge = context_gauge.ContextGauge()
        agent.run_sync(
            user_prompt="What modules are there?",
            model=FunctionModel(function=list_twice),
            deps=CodebaseDeps(root=TINYJOBS_ROOT),
            capabilities=[gauge, *extra],
        )
        last.append(gauge.readings[-1].used_tokens)
    whole, cut = last
    # Whole, the last request holds the responses, and the reading starts
    # from what they reported; cut to its last message, it holds none.
    assert whole > REPORTED_INPUT
    assert cut < REPORTED_INPUT // 10


def test_the_status_line_says_the_last_request_and_the_turn() -> None:
    from pydantic_ai_harness.compaction import ContextUsage

    from pydantic_tutor import context_gauge

    assumed = context_gauge.ContextGauge(
        readings=[
            ContextUsage(used_tokens=1_000, window_tokens=200_000, resolved=False),
            ContextUsage(used_tokens=3_156, window_tokens=200_000, resolved=False),
        ]
    )
    assert assumed.status_line() == (
        "[context 3,156 of 200,000 tokens (1.6%, window assumed) · "
        "this turn sent 4,156 tokens in 2 requests]"
    )
    known = context_gauge.ContextGauge(
        readings=[ContextUsage(used_tokens=232, window_tokens=400_000, resolved=True)]
    )
    assert known.status_line() == (
        "[context 232 of 400,000 tokens (0.1%) · this turn sent 232 tokens in 1 request]"
    )


def test_cost_session_shows_what_each_request_carried_and_the_gauge(
    project: Path,
) -> None:
    out = run_script(
        name=SESSION,
        cwd=project,
        stdin="What does tinyjobs.jobs do?\nAnd again?\n/exit\n",
    )
    assert out.startswith("Chatting with a stand-in model. Type /exit to leave.\n")
    carried = re.findall(
        pattern=r"request (\d+): instructions ([\d,]+) · "
        r"tool definitions ([\d,]+) \((\d+)\) · messages ([\d,]+) \((\d+)\)",
        string=out,
    )
    assert [int(found[0]) for found in carried] == [1, 2, 3, 4, 5, 6]
    # The instructions and the tool definitions are the same on every
    # request; the messages grow.
    assert len({found[1] for found in carried}) == 1
    assert len({(found[2], found[3]) for found in carried}) == 1
    sizes = [int(found[5]) for found in carried]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    lines = re.findall(
        pattern=r"\[context ([\d,]+) of 200,000 tokens \([\d.]+%, window assumed\) · "
        r"this turn sent ([\d,]+) tokens in 3 requests\]",
        string=out,
    )
    assert len(lines) == 2
    (context_1, sent_1), (context_2, sent_2) = (
        (int(a.replace(",", "")), int(b.replace(",", ""))) for a, b in lines
    )
    tools = int(carried[0][2].replace(",", ""))
    # By its last request, the context holds the tool definitions.
    assert context_1 > tools
    assert context_2 > context_1
    # Turn 1's first request had no count to start from, so turn 1 sent
    # less by about the tool definitions than turn 2.
    assert sent_2 - sent_1 > tools


def test_a_failed_turn_still_says_what_it_sent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pydantic_tutor.agent import agent
    from pydantic_tutor.chat import chat
    from pydantic_tutor.codebase_tools import CodebaseDeps
    from pydantic_tutor.composition import App

    def refuse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise UnexpectedModelBehavior(message="the provider refused")

    working_copy = tmp_path / "tinyjobs"
    shutil.copytree(src=TINYJOBS_ROOT, dst=working_copy)
    monkeypatch.setattr("builtins.input", Keyboard(lines=["hello", "/exit"]))
    chat(
        app=App(
            agent=agent,
            model=FunctionModel(function=refuse, model_name="stand-in"),
            deps=CodebaseDeps(root=working_copy),
        )
    )
    out = capsys.readouterr().out
    assert "UnexpectedModelBehavior: the provider refused\n" in out
    assert re.search(
        pattern=r"\[context [\d,]+ of 200,000 tokens \([\d.]+%, window assumed\) · "
        r"this turn sent [\d,]+ tokens in 1 request\]\n",
        string=out,
    )
