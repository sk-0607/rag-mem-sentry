import json
from dataclasses import dataclass

from gate.signals.judge import _UNTRUSTED_RULE, _strip_fence, _user_prompt
from gate.types import Item


class CorroborationResponseError(ValueError):
    """Raised when the model's reply violates the extraction contract."""


@dataclass(frozen=True)
class Claim:
    text: str
    span: str | None


# Braces in the JSON shape are doubled: the prompt is filled with str.format
# so S3_MAX_CLAIMS is read at call time, not frozen at import.
_EXTRACT_SYSTEM = (
    "You extract checkable factual claims from passages. For each item, list "
    "up to {max_claims} claims. A checkable claim is a statement a reference "
    "text could confirm or contradict: a number, a date, a name, an "
    "attribution, a definition, a stated result. Opinions, instructions and "
    "questions are not claims. Write each claim as one self-contained "
    "sentence with pronouns resolved. `span` is the shortest verbatim passage "
    "of the item the claim came from.\n\n"
    + _UNTRUSTED_RULE
    + "\n\n"
    "Respond with ONLY a JSON array, one object per item, in this exact "
    'shape: [{{"id": "i0", "claims": [{{"text": "<sentence>", "span": '
    '"<verbatim passage>"}}]}}, ...]. No prose outside the array.'
)


def _parse_claims(raw: str, items: list[Item], max_claims: int) -> list[list[Claim]]:
    if not isinstance(raw, str):
        raise CorroborationResponseError(f"reply was {type(raw).__name__}, not text")
    try:
        parsed = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as e:
        raise CorroborationResponseError(f"response was not valid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise CorroborationResponseError(f"expected a JSON array, got {type(parsed).__name__}")

    expected = [f"i{idx}" for idx in range(len(items))]
    by_id: dict[str, dict] = {}
    for entry in parsed:
        if not isinstance(entry, dict) or "id" not in entry:
            raise CorroborationResponseError(f"malformed entry: {entry!r}")
        entry_id = entry["id"]
        if entry_id not in expected:
            raise CorroborationResponseError(f"unknown id {entry_id!r} in response")
        if entry_id in by_id:
            raise CorroborationResponseError(f"duplicate id {entry_id!r} in response")
        by_id[entry_id] = entry
    missing = [eid for eid in expected if eid not in by_id]
    if missing:
        raise CorroborationResponseError(f"missing id(s) in response: {missing}")

    out = []
    for idx, item in enumerate(items):
        claims = by_id[f"i{idx}"].get("claims")
        if not isinstance(claims, list):
            raise CorroborationResponseError(f"claims must be a list for i{idx}")
        kept = []
        for c in claims:
            text = c.get("text") if isinstance(c, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise CorroborationResponseError(f"claim without text for i{idx}: {c!r}")
            # Non-verbatim span is dropped, claim kept — same call judge.py makes.
            span = c.get("span")
            if not isinstance(span, str) or span not in item.text:
                span = None
            kept.append(Claim(text=text, span=span))
        # Over-long list is truncated, not an error: the first claims are the
        # ones the model ranked most salient, and a cap is policy, not contract.
        out.append(kept[:max_claims])
    return out


class CorroborationSignal:
    """S3: checks an item's factual claims against a trusted reference index."""

    name = "corroborate"

    def __init__(self, reference, chat=None, model=None):
        self._reference = reference
        self._chat = chat
        self._model = model

    def extract_claims(self, items: list[Item]) -> list[list[Claim]]:
        if not items:
            return []
        # Config and chat resolved at call time so a monkeypatch after
        # construction takes effect, and gate/ imports no app/ at import time.
        from app import config
        import app.llm

        chat = self._chat if self._chat is not None else app.llm.chat
        model = self._model if self._model is not None else config.JUDGE_MODEL
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM.format(max_claims=config.S3_MAX_CLAIMS)},
            {"role": "user", "content": _user_prompt(items)},
        ]
        return _parse_claims(chat(messages, model=model), items, config.S3_MAX_CLAIMS)
