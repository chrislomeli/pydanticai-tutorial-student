"""The color helper colors a terminal, and nothing else (02-02)."""

import io
import sys

import pytest

from pydantic_tutor.color import blue, dim


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_text_is_plain_when_output_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert blue(text="agent> hi") == "agent> hi"
    assert dim(text="(tokens)") == "(tokens)"


def test_text_is_colored_on_a_terminal_and_reset_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", _Terminal())
    assert blue(text="hi") == "\033[38;5;75mhi\033[0m"
    assert dim(text="hi") == "\033[2mhi\033[0m"


def test_no_color_turns_colors_off_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "stdout", _Terminal())
    assert blue(text="hi") == "hi"
    assert dim(text="hi") == "hi"
