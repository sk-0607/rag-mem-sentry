import re

from gate.types import GateContext, Item, SignalResult

# (label, pattern). Labels appear in the reason string so a reviewer reading
# the audit log knows which rule fired without re-running anything.
INSTRUCTION_PATTERNS: list[tuple[str, str]] = [
    ("ignore-previous", r"ignore\s+(?:all\s+)?(?:previous|prior|earlier|above)\s+instructions?"),
    ("disregard-above", r"disregard\s+(?:all\s+)?(?:previous|prior|earlier|the\s+above)"),
    ("role-reassignment", r"you\s+are\s+now\s+(?:an?\s+)?\w+"),
    ("fake-system-message", r"system\s+(?:prompt|message)\s*:"),
    ("new-instructions", r"new\s+instructions?\s*:"),
]

_COMPILED = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in INSTRUCTION_PATTERNS]

# Weight per distinct matched pattern, combined by noisy-OR within the signal.
# 0.45 keeps a single match below the hard threshold on purpose: the corpus
# quotes real attacks, and one quotation must quarantine rather than block.
PATTERN_WEIGHT = 0.45

OBFUSCATION_CHECKS: list[tuple[str, str]] = [
    ("zero-width", r"[​‌‍⁠﻿]"),
    ("bidi-override", r"[‪-‮⁦-⁩]"),
    ("hidden-css", r"(?:display\s*:\s*none|font-size\s*:\s*0(?!\.\d)(?!\d)(?:px|pt)?|visibility\s*:\s*hidden)"),
    ("html-comment", r"<!--.*?-->"),
]

_COMPILED_OBFUSCATION = [
    (label, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for label, pattern in OBFUSCATION_CHECKS
]

# Obfuscation weighs heavier than an instruction pattern. Academic prose quotes
# attack text; it does not contain zero-width characters or bidi overrides.
OBFUSCATION_WEIGHT = 0.8


class StructuralSignal:
    """Rules-only detection over raw content. No LLM, no network."""

    name = "structural"
    # The hard-block threshold for this signal lives in gate/policy.py's
    # DEFAULT_POLICY.hard_thresholds["structural"] (0.93) — that's the value
    # GatePolicy.decide() actually reads. See the comment there for how it
    # was derived from the weights above.

    def score(self, item: Item, ctx: GateContext) -> SignalResult | None:
        hits: list[tuple[str, str, float]] = []

        for label, pattern in _COMPILED:
            match = pattern.search(item.text)
            if match:
                hits.append((label, match.group(0), PATTERN_WEIGHT))

        for label, pattern in _COMPILED_OBFUSCATION:
            match = pattern.search(item.text)
            if match:
                hits.append((label, match.group(0), OBFUSCATION_WEIGHT))

        if not hits:
            return None

        clean = 1.0
        for _, _, weight in hits:
            clean *= 1.0 - weight
        suspicion = min(1.0, 1.0 - clean)

        labels = ", ".join(label for label, _, _ in hits)
        return SignalResult(
            signal=self.name,
            score=suspicion,
            reason=f"matched {len(hits)} rule(s): {labels}",
            evidence_span=hits[0][1],
        )
