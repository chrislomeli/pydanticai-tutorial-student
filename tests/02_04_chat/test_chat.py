"""Checks that the project from the chat lesson is in place.

Our checks for the chat lesson: they verify the lesson, and the student is
not asked to run them. Run them with `uv run pytest`.

All four of this lesson's tests were moved by the settings lesson
(03-01), which takes `DEFAULT_MODEL`, `TRACE_VAR` and `configure_tracing`
out of `__main__.py`: the model and the tracing flag became settings, and
tracing is wired in the composition root. Their points are now proved in
`tests/03_01_settings/test_settings.py`:

- tracing off without the flag, and spans for a scripted turn with it:
  `test_tracing_stays_off_without_the_setting` and
  `test_tracing_prints_spans_for_a_scripted_turn`;
- a missing key fails by name: `test_a_missing_key_fails_when_the_app_is_built`,
  and the entry point's `test_the_entry_point_stops_by_name_before_the_chat_opens`;
- the entry point uses the project's one agent, and defaults to a real
  provider: `test_build_app_builds_the_objects_from_the_settings` and
  `test_settings_have_the_defaults_the_chat_used`.
"""
