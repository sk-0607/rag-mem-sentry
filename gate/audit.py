import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gate.types import Item, Verdict


class AuditLog:
    """Append-only record of every gate verdict, admits included.

    This is both the demo surface and the evaluation data source, so nothing
    is filtered out at write time.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, item: Item, verdict: Verdict, store: str, stage: str = "write") -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "item_id": item.id,
            "store": store,
            "stage": stage,
            "action": verdict.action.value,
            "trust": verdict.score.trust,
            "reason": verdict.reason,
            "signals": [asdict(r) for r in verdict.score.results],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
