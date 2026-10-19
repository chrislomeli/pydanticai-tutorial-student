"""Checks that the project from the typed-answer lesson is in place.

Our checks for the typed-answer lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/01_06_typed_answer/test_typed_answer.py: two folders
# below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
INVOICES: Path = PROJECT_ROOT / "sample_docs" / "invoices"

GOOD: dict[str, object] = {
    "number": "INV-0042",
    "issued": "2026-09-30",
    "supplier": "Acme Paper Co.",
    "currency": "GBP",
    "lines": [
        {"description": "A4 paper, box of 5", "quantity": 3, "unit_price": "24.50"},
        {"description": "Stapler", "quantity": 1, "unit_price": "12.00"},
    ],
    "total": "85.50",
}
WRONG: dict[str, object] = {**GOOD, "total": "73.50"}


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the lesson's scripts, by its path from the project root; return its stdout."""
    result = subprocess.run(
        args=[sys.executable, name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # The banner goes to stderr; turning it off keeps a failure readable.
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


class Scripted:
    """A model that answers with the next of `answers`: a dict is a call to the output tool, a str is text."""

    def __init__(self, *answers: dict[str, object] | str) -> None:
        self.answers = list(answers)
        self.infos: list[AgentInfo] = []
        self.seen: list[list[ModelMessage]] = []

    def reply(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.infos.append(info)
        self.seen.append(list(messages))
        answer = self.answers.pop(0)
        if isinstance(answer, str):
            return ModelResponse(parts=[TextPart(content=answer)])
        name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=answer)])


def test_the_documents_are_in_the_project() -> None:
    names = sorted(path.name for path in INVOICES.glob("*.txt"))
    assert names == ["INV-0042.txt", "INV-0043.txt", "NW-2026-117.txt"]


def test_the_model_is_given_a_tool_to_answer_with_and_no_text() -> None:
    from pydantic_tutor.invoices import extractor

    model = Scripted(GOOD)
    extractor.run_sync(user_prompt="x", model=FunctionModel(function=model.reply))

    info = model.infos[0]
    assert [tool.name for tool in info.output_tools] == ["final_result"]
    assert info.function_tools == []
    assert info.allow_text_output is False
    # The model is told what each field means: the descriptions are sent.
    schema = info.output_tools[0].parameters_json_schema
    assert schema["properties"]["total"]["description"]
    assert sorted(schema["required"]) == [
        "currency",
        "issued",
        "lines",
        "number",
        "supplier",
        "total",
    ]


def test_a_correct_answer_comes_back_as_an_invoice() -> None:
    from pydantic_tutor.invoices import Invoice, extractor

    model = Scripted(GOOD)
    result = extractor.run_sync(
        user_prompt="x", model=FunctionModel(function=model.reply)
    )

    invoice = result.output
    assert isinstance(invoice, Invoice)
    assert invoice.issued == date(year=2026, month=9, day=30)
    assert invoice.total == Decimal("85.50")
    assert isinstance(invoice.lines[0].unit_price, Decimal)
    assert result.usage.requests == 1


def test_the_invoice_refuses_a_total_that_is_not_the_sum() -> None:
    from pydantic_tutor.invoices import Invoice

    # The rule belongs to the data: it holds wherever an Invoice is made.
    with pytest.raises(expected_exception=ValidationError, match=r"85\.50"):
        Invoice.model_validate(obj=WRONG)


def test_a_wrong_total_is_sent_back_and_corrected() -> None:
    from pydantic_tutor.invoices import extractor

    model = Scripted(WRONG, GOOD)
    result = extractor.run_sync(
        user_prompt="x", model=FunctionModel(function=model.reply)
    )

    assert result.output.total == Decimal("85.50")
    assert result.usage.requests == 2
    [part] = model.seen[1][-1].parts
    assert isinstance(part, RetryPromptPart)
    assert "is not the sum of the lines, 85.50" in part.model_response()


def test_a_total_wrong_twice_ends_the_run() -> None:
    from pydantic_tutor.invoices import extractor

    with pytest.raises(
        expected_exception=UnexpectedModelBehavior,
        match=r"Exceeded maximum output retries \(1\)",
    ):
        extractor.run_sync(
            user_prompt="x",
            model=FunctionModel(function=Scripted(WRONG, WRONG).reply),
        )


def test_extract_invoice_shows_the_retry_and_the_corrected_invoice() -> None:
    printed = run_script("scripts/extract_invoice.py")

    lines = printed.splitlines()
    assert lines[0] == "the model answers by calling: final_result"
    assert "sent back to the model:" in lines
    assert "is not the sum of the lines, 85.50" in printed
    assert lines[-4:] == [
        "number: INV-0042",
        "issued: 2026-09-30",
        "total: 85.50 GBP",
        "requests: 2",
    ]
