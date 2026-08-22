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
    (
        "hidden-css",
        r"(?:display\s*:\s*none"
        r"|font-size\s*:\s*0(?!\.\d)(?!\d)(?:px|pt)?"
        r"|visibility\s*:\s*hidden"
        r"|opacity\s*:\s*0(?![\d.])"
        r"|(?<![\w-])color\s*:\s*transparent"
        r"|clip-path\s*:\s*inset\(\s*100%\s*\)"
        r"|(?<![\w-])height\s*:\s*0(?![\d.])(?:px|pt|%)?"
        r"|(?<![\w-])width\s*:\s*0(?![\d.])(?:px|pt|%)?)",
    ),
    ("html-comment", r"<!--.*?-->"),
]

_COMPILED_OBFUSCATION = [
    (label, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for label, pattern in OBFUSCATION_CHECKS
]

# Obfuscation weighs heavier than an instruction pattern. Academic prose quotes
# attack text; it does not contain zero-width characters or bidi overrides.
OBFUSCATION_WEIGHT = 0.8

# Confusable non-ASCII codepoints that actually appear in this attack, mapped
# to their ASCII lookalikes. Explicit dict beats pulling in a `confusables`
# dependency for a dozen characters.
CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic look-alikes for Latin letters.
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    # Greek look-alikes.
    "ο": "o", "ν": "v", "ρ": "p",
}
# Fullwidth Latin forms (U+FF21-FF3A, U+FF41-FF5A) map onto plain ASCII.
for _i in range(26):
    CONFUSABLE_MAP[chr(0xFF21 + _i)] = chr(ord("A") + _i)
    CONFUSABLE_MAP[chr(0xFF41 + _i)] = chr(ord("a") + _i)

_SKELETON_TABLE = str.maketrans(CONFUSABLE_MAP)


def _skeleton(text: str) -> str:
    """Map confusable codepoints to their ASCII lookalikes.

    Skeletonizing, not translating: unmapped non-ASCII characters (accents,
    genuine Cyrillic/Greek prose) pass through untouched, so this only ever
    *reveals* a Latin-script instruction pattern hiding behind homoglyphs —
    it never itself flags plain non-English text.
    """
    return text.translate(_SKELETON_TABLE)


class StructuralSignal:
    """Rules-only detection over raw content. No LLM, no network."""

    name = "structural"
    # The hard-block threshold for this signal lives in gate/policy.py's
    # DEFAULT_POLICY.hard_thresholds["structural"] (0.93) — that's the value
    # GatePolicy.decide() actually reads. See the comment there for how it
    # was derived from the weights above.

    def score(self, item: Item, ctx: GateContext) -> SignalResult | None:
        hits: list[tuple[str, str, float]] = []
        raw_matched_labels: set[str] = set()

        for label, pattern in _COMPILED:
            match = pattern.search(item.text)
            if match:
                hits.append((label, match.group(0), PATTERN_WEIGHT))
                raw_matched_labels.add(label)

        for label, pattern in _COMPILED_OBFUSCATION:
            match = pattern.search(item.text)
            if match:
                hits.append((label, match.group(0), OBFUSCATION_WEIGHT))

        # Skeletonize and re-run the instruction patterns. A match that only
        # appears after mapping homoglyphs to ASCII is stronger evidence than
        # a plain match — nobody types Cyrillic а inside an English imperative
        # by accident — so it is weighed at OBFUSCATION_WEIGHT, not
        # PATTERN_WEIGHT. Patterns that already matched the raw text are
        # skipped: skeletonizing must only ever *raise* a new hit, never
        # double-count one that plain text already produced.
        skeleton = _skeleton(item.text)
        if skeleton != item.text:
            for label, pattern in _COMPILED:
                if label in raw_matched_labels:
                    continue
                match = pattern.search(skeleton)
                if match:
                    hits.append((f"homoglyph-{label}", match.group(0), OBFUSCATION_WEIGHT))

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
