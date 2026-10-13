import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from gate.cache import content_hash

# Local tokenizer: importing app.retrieval would load the host's models into gate/.
_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class ReferenceIndex:
    """Owner-trusted chunks only. `trusted is True` is the sole admission test:
    a truthy string like "true" must not slip in."""

    def __init__(self, chunks: list[dict]):
        self._chunks = [c for c in chunks if c.get("trusted") is True]
        self._tokens = [_tokens(c["text"]) for c in self._chunks]
        self._texts = {c["text"] for c in self._chunks}
        # BM25Okapi divides by corpus size and fails on an empty corpus.
        self._bm25 = BM25Okapi(self._tokens) if self._chunks else None

    def search(self, claim: str, k: int) -> list[dict]:
        if k <= 0 or self._bm25 is None:
            return []
        query = _tokens(claim)
        scores = self._bm25.get_scores(query)
        qset = set(query)
        # Relevance is shared-token overlap, not score sign: IDF goes
        # non-positive for a term in over half of a small corpus.
        hits = [(scores[i], c["id"], c) for i, c in enumerate(self._chunks)
                if qset & set(self._tokens[i])]
        hits.sort(key=lambda h: (-h[0], h[1]))
        return [c for _, _, c in hits[:k]]

    def contains_text(self, text: str) -> bool:
        return text in self._texts

    def fingerprint(self) -> str:
        lines = sorted(f"{c['id']}:{content_hash(c['text'])}" for c in self._chunks)
        return content_hash("\n".join(lines))

    def __len__(self) -> int:
        return len(self._chunks)


def default_reference(data_dir: Path) -> ReferenceIndex:
    path = Path(data_dir) / "chunks.jsonl"
    if not path.exists():
        return ReferenceIndex([])
    with path.open(encoding="utf-8") as f:
        return ReferenceIndex([json.loads(line) for line in f if line.strip()])
