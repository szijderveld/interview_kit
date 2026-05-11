"""Resume acknowledgement template — Step 10 / D9.

When ``run_loop`` finds a saved :class:`SessionRuntimeState` on entry,
the first agent utterance after rehydration is the constant below. Its
wording is fixed and known-good, so the runner intentionally bypasses
:func:`validate_voice_phrasing` — the constant is the spec, not a
candidate the model produced.

Per-Conversation customisation is deferred to v2 (SCOPE open question 6).
"""

from __future__ import annotations

RESUME_ACK = "we got cut off — let me pick up where we left off."
