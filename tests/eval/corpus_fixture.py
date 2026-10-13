"""Synthetic attack corpus for testing the eval tooling. Holds no attack content."""
import json
from pathlib import Path


def make_items(per_class: int = 30, class2_groups: int = 10) -> list[dict]:
    items = []
    for cls in (1, 2, 3):
        for i in range(per_class):
            # Class 2 must span class2_groups groups; the others group in threes.
            group = f"c2-g{i % class2_groups}" if cls == 2 else f"c{cls}-g{i // 3}"
            item = {
                "id": f"c{cls}-{i}", "class": cls, "group": group,
                "target_question": f"FIXTURE question {cls}-{i}?",
                "check": f"FIXTURE check {cls}-{i}",
            }
            if cls in (1, 2):
                item["text"] = f"FIXTURE text {cls}-{i}"
            if cls == 2:
                item["target_answer"] = f"FIXTURE answer {cls}-{i}"
            if cls == 3:
                item.update(task=f"FIXTURE task {i}", approach=f"FIXTURE approach {i}",
                            outcome=f"FIXTURE outcome {i}")
            items.append(item)
    return items


def write_corpus(directory: Path, items: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "items.jsonl"
    path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    return path
