"""Checks that the project from the multimodal lesson is in place.

Our checks for the multimodal lesson: they verify the lesson, and the
student is not asked to run them. Run them with `uv run pytest`. They fail
until the lesson is done.

Every import of the project is inside the test that needs it, so that
before the lesson each test fails on its own, instead of the whole file
failing to load.
"""

import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# This file is tests/02_05_multimodal/test_multimodal.py: two folders
# below the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCAN: Path = PROJECT_ROOT / "sample_docs" / "invoices" / "INV-0042-scan.png"

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


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any request to a real model raises instead of reaching a provider.
    monkeypatch.setattr(target=models, name="ALLOW_MODEL_REQUESTS", value=False)


def run_script(name: str) -> str:
    """Run one of the lesson's examples, by its path from the project root; return its stdout."""
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


def test_the_scan_is_in_the_project_and_is_a_real_image() -> None:
    scan = BinaryContent.from_path(SCAN)
    assert scan.is_image
    assert scan.media_type == "image/png"
    assert len(scan.data) > 0


def test_the_extractor_reads_the_scan_with_no_code_change() -> None:
    from pydantic_tutor.invoices import Invoice, extractor

    scan = BinaryContent.from_path(SCAN)
    seen: list[AgentInfo] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=GOOD)])

    result = extractor.run_sync(
        user_prompt=["Pull the total out of this scan.", scan],
        model=FunctionModel(function=reply),
    )

    # Same shape as 01-06-typed-answer's test: nothing about the agent
    # changed because its input is now an image.
    info = seen[0]
    assert [tool.name for tool in info.output_tools] == ["final_result"]
    assert info.function_tools == []
    assert info.allow_text_output is False

    invoice = result.output
    assert isinstance(invoice, Invoice)
    assert invoice.total == Decimal("85.50")
    assert result.usage.requests == 1


def test_the_same_construction_works_for_a_document() -> None:
    from pydantic_tutor.invoices import Invoice, extractor

    # No PDF fixture file ships: invoices.py draws no distinction between
    # an image and a document, so a document built from bytes proves the
    # same mechanism without a second fixture (plan.md §4).
    document = BinaryContent(data=b"%PDF-1.4\n", media_type="application/pdf")
    assert document.is_document

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=GOOD)])

    result = extractor.run_sync(
        user_prompt=["Pull the total out of this scan.", document],
        model=FunctionModel(function=reply),
    )

    invoice = result.output
    assert isinstance(invoice, Invoice)
    assert invoice.total == Decimal("85.50")


def test_extract_scan_prints_the_invoice() -> None:
    printed = run_script("examples/02-05-multimodal/extract_scan.py")

    assert printed.splitlines() == [
        "number: INV-0042",
        "total: 85.50 GBP",
        "requests: 1",
    ]
