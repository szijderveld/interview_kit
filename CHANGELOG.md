# Changelog

## 0.3.0 (unreleased)

- **Breaking:** `EvalResult.next_action` no longer accepts `"drill"`;
  use `"probe"` together with a required `probe_kind` (one of
  `clarify`, `example`, `importance`, `contrast`, `elaborate`). The
  compose prompt branches on `probe_kind` to ask materially different
  follow-ups. `retry` is unchanged.

## 0.2.0

- Renamed distribution and import to `interview_kit`.
- Added PyPI metadata, console script, and `[voice]` extra.
- README rewritten around `pip install`; `CONTRIBUTING.md` split out.

## 0.1.0

Initial release.
