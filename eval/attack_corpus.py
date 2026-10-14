"""Validate, freeze, split and load the external attack corpus (spec section 8).

The corpus is delivered from outside the repo. Freezing hashes it so a later edit
cannot silently move the numbers measured against it.
"""
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

from gate.types import Item
from memory.types import ExperienceRecord

MIN_PER_CLASS = 30
MIN_CLASS2_GROUPS = 10
# Adaptive sets are smaller: 60 probes of one bit buy fewer finished items than
# a blind author writes (Phase 5 spec §10.1).
ADAPTIVE_MIN_PER_CLASS = 15
ADAPTIVE_MIN_CLASS2_GROUPS = 5
SPLITS = ("calibration", "test")
ADAPTIVE = "adaptive"
ITEMS_FILE = "items.jsonl"
MANIFEST_FILE = "manifest.json"
REQUIRED = {
    1: ("text",),
    2: ("text", "target_answer"),
    3: ("task", "approach", "outcome"),
}
COMMON = ("target_question", "check")
# Fixed so the screened text, and so the gate cache key, is identical on every run.
_FIXED_TS = "1970-01-01T00:00:00+00:00"


class AttackCorpusError(ValueError):
    pass


def corpus_dir() -> Path:
    return Path(os.environ.get("ATTACK_CORPUS_DIR", "data/attack_corpus"))


def read_items(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AttackCorpusError(f"line {n}: not valid JSON ({e.msg})") from e
            if not isinstance(obj, dict):
                raise AttackCorpusError(f"line {n}: not a JSON object")
            items.append(obj)
    return items


def _nonempty(v) -> bool:
    return isinstance(v, str) and v != ""


def validate(items: list[dict], adaptive: bool = False) -> None:
    errors = []
    seen = set()
    group_classes: dict[str, set] = {}
    per_class = {1: 0, 2: 0, 3: 0}
    class2_groups = set()
    for n, it in enumerate(items, 1):
        iid = it.get("id")
        if not _nonempty(iid):
            errors.append(f"item #{n}: id missing or not a non-empty string")
            iid = f"#{n}"
        elif iid in seen:
            errors.append(f"{iid}: duplicate id")
        seen.add(iid)
        if "trusted" in it:
            errors.append(f"{iid}: 'trusted' key not allowed; attack items are always untrusted")
        cls = it.get("class")
        # type check, not `in`: True == 1 would pass an `in (1, 2, 3)` test.
        if type(cls) is not int or cls not in REQUIRED:
            errors.append(f"{iid}: class must be the integer 1, 2 or 3, got {cls!r}")
            continue
        per_class[cls] += 1
        for field in COMMON + ("group",) + REQUIRED[cls]:
            if not _nonempty(it.get(field)):
                errors.append(f"{iid}: field '{field}' missing or not a non-empty string")
        group = it.get("group")
        if _nonempty(group):
            group_classes.setdefault(group, set()).add(cls)
            if cls == 2:
                class2_groups.add(group)
        ans, q = it.get("target_answer"), it.get("target_question")
        if cls == 2 and _nonempty(ans) and _nonempty(q) and ans.casefold() in q.casefold():
            errors.append(f"{iid}: target_answer occurs in its own target_question")
    for group, classes in sorted(group_classes.items()):
        if len(classes) > 1:
            errors.append(f"group {group}: spans classes {sorted(classes)}")
    min_items = ADAPTIVE_MIN_PER_CLASS if adaptive else MIN_PER_CLASS
    min_groups = ADAPTIVE_MIN_CLASS2_GROUPS if adaptive else MIN_CLASS2_GROUPS
    for cls, n in per_class.items():
        if n < min_items:
            errors.append(f"class {cls}: {n} items, need at least {min_items}")
    if len(class2_groups) < min_groups:
        errors.append(f"class 2: {len(class2_groups)} groups, need at least {min_groups}")
    if errors:
        raise AttackCorpusError("\n".join(errors))


def assign_splits(items: list[dict]) -> dict[str, str]:
    by_class: dict[int, set] = {}
    for it in items:
        by_class.setdefault(it["class"], set()).add(it["group"])
    splits = {}
    for groups in by_class.values():
        # Hash order, not file order or sorted ids: the result must depend on ids alone.
        ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode("utf-8")).hexdigest())
        for i, g in enumerate(ordered):
            splits[g] = SPLITS[i % 2]
    return splits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_kind(manifest: dict) -> str:
    # A manifest frozen before Phase 5 has no kind key; it is blind.
    return manifest.get("kind", "blind")


def freeze(directory: Path, today: str, adaptive: bool = False) -> dict:
    path = directory / ITEMS_FILE
    items = read_items(path)
    validate(items, adaptive=adaptive)
    if adaptive:
        groups = {it["group"]: ADAPTIVE for it in items}
        splits = (ADAPTIVE,)
    else:
        groups, splits = assign_splits(items), SPLITS
    counts = {str(c): {s: 0 for s in splits} for c in REQUIRED}
    for it in items:
        counts[str(it["class"])][groups[it["group"]]] += 1
    manifest = {"sha256": _sha256(path), "frozen": today, "counts": counts, "groups": groups,
                "kind": ADAPTIVE if adaptive else "blind"}
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(directory: Path, split: str | None = None) -> list[dict]:
    if split is not None and split not in SPLITS + (ADAPTIVE,):
        raise AttackCorpusError(f"unknown split {split!r}; use 'calibration', 'test', 'adaptive' or None")
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise AttackCorpusError(f"corpus not frozen: no {MANIFEST_FILE} in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = manifest_kind(manifest)
    allowed = (ADAPTIVE,) if kind == ADAPTIVE else SPLITS
    if split is not None and split not in allowed:
        # A blind split read from an adaptive set (or the reverse) would score one
        # attacker setting as the other.
        raise AttackCorpusError(f"split {split!r} is not valid on a {kind} corpus")
    path = directory / ITEMS_FILE
    if _sha256(path) != manifest["sha256"]:
        raise AttackCorpusError(f"{ITEMS_FILE} changed since freeze")
    items = []
    for it in read_items(path):
        it["split"] = manifest["groups"][it["group"]]
        if split is None or it["split"] == split:
            items.append(it)
    return items


def to_experience_record(entry: dict) -> ExperienceRecord:
    # One projection for the screen and the demo plant: the item text is in the
    # verdict cache key, so a second copy that drifted would miss the cache.
    return ExperienceRecord(id=entry["id"], task=entry["task"], approach=entry["approach"],
                            outcome=entry["outcome"], success=False, ts=_FIXED_TS)


def to_gate_item(entry: dict) -> tuple[Item, str]:
    if entry["class"] in (1, 2):
        return Item(id=entry["id"], text=entry["text"], kind="corpus",
                    metadata={"trusted": False}), "corpus"
    return to_experience_record(entry).to_item(), "experience"


def main(argv: list[str]) -> int:
    adaptive = argv[1:] == ["--adaptive"]
    if not argv or argv[0] not in ("validate", "freeze") or (len(argv) > 1 and not adaptive):
        print("usage: python -m eval.attack_corpus validate|freeze [--adaptive]", file=sys.stderr)
        return 2
    directory = corpus_dir()
    try:
        if argv[0] == "validate":
            validate(read_items(directory / ITEMS_FILE), adaptive=adaptive)
            print(f"{directory}: valid")
        else:
            freeze(directory, date.today().isoformat(), adaptive=adaptive)
            print(f"{directory}: frozen")
    except AttackCorpusError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
