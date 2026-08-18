import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from gate.types import SignalResult

log = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """Stable content key. Exact text, no normalisation — a smuggled zero-width
    character makes it a different item, which is the correct behaviour."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VerdictCache:
    """Append-only JSONL cache of signal results, keyed by content hash.

    Loaded into memory on open; the last line written for a key wins.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, list[SignalResult]] = {}
        self._handle = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                self._entries[record["key"]] = [SignalResult(**r) for r in record["results"]]
            except Exception:
                log.warning("skipping corrupt cache line in %s", self.path)

    def get(self, key: str) -> list[SignalResult] | None:
        return self._entries.get(key)

    def put(self, key: str, results: list[SignalResult]) -> None:
        self._entries[key] = list(results)
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps({"key": key, "results": [asdict(r) for r in results]}) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
