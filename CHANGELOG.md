# Changelog

## 0.3.0 (unreleased)

- **Breaking:** `EvalResult.next_action` no longer accepts `"drill"`;
  use `"probe"` together with a required `probe_kind` (one of
  `clarify`, `example`, `importance`, `contrast`, `elaborate`). The
  compose prompt branches on `probe_kind` to ask materially different
  follow-ups. `retry` is unchanged.
- Added `EvalResult.clarity` (`clear` / `hedged` / `vague`, default
  `clear`). The runner now overrides a non-resolving `next_action` to
  a `probe_clarify` when the model reports a hedged or vague answer,
  capped at once per goal. Internal loop signal only — not surfaced
  on `GoalStatus` or `Extract`.
- Refusal and IDK no longer share a code path. A consent-decline
  ("I'd rather not", "prefer not", …) marks the active goal
  `skipped_refused` (new `GoalStatusValue`) on the first hit and
  advances without speaking a deflection — pressing on a stated
  boundary is wrong. "I don't know" keeps the two-strike behavior:
  one deflection probe, then `gave_up`. Refused goals are resolved
  for the rest of the session and are never re-selected.

## 0.2.0

- Renamed distribution and import to `interview_kit`.
- Added PyPI metadata, console script, and `[voice]` extra.
- README rewritten around `pip install`; `CONTRIBUTING.md` split out.

## 0.1.0

Initial release.
