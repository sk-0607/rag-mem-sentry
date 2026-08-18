import json
from pathlib import Path

from gate.types import Item, Verdict


class Quarantine:
    """Items withheld from a store but not destroyed. Releasable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._items: dict[str, Item] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["op"] == "hold":
                self._items[record["item"]["id"]] = Item(**record["item"])
            elif record["op"] == "release":
                self._items.pop(record["item_id"], None)

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def hold(self, item: Item, verdict: Verdict) -> None:
        self._items[item.id] = item
        self._append({
            "op": "hold",
            "item": {"id": item.id, "text": item.text, "kind": item.kind, "metadata": item.metadata},
            "reason": verdict.reason,
        })

    def held(self) -> list[Item]:
        return list(self._items.values())

    def release(self, item_id: str) -> Item:
        item = self._items.pop(item_id)
        self._append({"op": "release", "item_id": item_id})
        return item
