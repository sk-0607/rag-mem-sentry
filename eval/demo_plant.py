"""Plant one blind class-3 record into the demo's experience store, and purge it after.

    python -m eval.demo_plant candidates          # test-split class-3 items S2 excluded in Task 46
    python -m eval.demo_plant plant --item <id>   # through the real write gate; no LLM call
    python -m eval.demo_plant purge [--yes]       # without --yes, lists what would go

The plant goes through `ExperienceStore.write(gate=)`, the same write gate every
record passes, so a record the write gate would not admit is never planted.
Purge removes the planted record and every record written after it (the demo's
own runs), after copying the mirror to a backup.
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.attack_corpus import corpus_dir, load, to_experience_record
from eval.freeze_gate import FreezeError, require_frozen
from gate.types import Action

PLANT_FILE = Path("data/demo_plant.json")
# Task 46's test-split screen, at the frozen state: the candidates are items it
# has already seen S2 exclude, so the demo shows a known verdict.
SCREEN_RESULTS = Path("data/phase4_screen/test/results.json")


class PlantError(RuntimeError):
    pass


def load_screen_rows(path: Path = SCREEN_RESULTS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise PlantError(f"no screen results at {path}: run the Task 46 test-split screen first")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def candidates(entries: list[dict], screen_rows: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in entries if e["class"] == 3}
    return [{"id": r["id"], "target_question": by_id[r["id"]]["target_question"]}
            for r in screen_rows
            if r["id"] in by_id and r["action"] != "admit" and r["top_signal"] == "judge"
            and not r.get("reduced_assurance")]


def _now() -> str:
    # The format record_from_run mints `ts` in, so purge_ids can compare strings.
    return datetime.now(timezone.utc).isoformat()


def plant(item_id: str, entries: list[dict], store, gate, plant_file: Path = PLANT_FILE,
          now=_now) -> dict:
    require_frozen()
    plant_file = Path(plant_file)
    if plant_file.exists():
        raise PlantError(f"{plant_file} exists: purge the earlier plant first")
    entry = next((e for e in entries if e["id"] == item_id), None)
    if entry is None:
        raise PlantError(f"{item_id!r} is not in the test split")
    if entry["class"] != 3:
        raise PlantError(f"{item_id!r} is class {entry['class']}, not class 3")
    record = to_experience_record(entry)
    planted_ts = now()
    verdict = store.write(record, gate=gate)
    # The store persists only on admit, so a refusal here has written nothing.
    if verdict is None or verdict.action is not Action.ADMIT:
        action = verdict.action.value if verdict is not None else "no verdict"
        reason = verdict.reason if verdict is not None else ""
        raise PlantError(f"write gate did not admit {item_id!r}: {action} ({reason})")
    out = {"item_id": item_id, "record_id": record.id, "planted_ts": planted_ts,
           "target_question": entry["target_question"]}
    plant_file.parent.mkdir(parents=True, exist_ok=True)
    plant_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def purge_ids(plant: dict, records) -> list[str]:
    # The planted record keeps the corpus projection's fixed ts, so it is named
    # by id; the demo's own records are the ones written at or after the plant.
    ids = []
    for r in records:
        if (r.id == plant["record_id"] or r.ts >= plant["planted_ts"]) and r.id not in ids:
            ids.append(r.id)
    return ids


def purge(store, mirror_path: Path, plant_file: Path = PLANT_FILE, yes: bool = False) -> list[str]:
    plant_file = Path(plant_file)
    if not plant_file.exists():
        raise PlantError("nothing planted")
    plant = json.loads(plant_file.read_text(encoding="utf-8"))
    ids = purge_ids(plant, store.all_records())
    if not yes:
        return ids
    mirror_path = Path(mirror_path)
    # Backup first: purge is destructive, and the demo's records are the only copy.
    backup = mirror_path.with_name(f"{mirror_path.name}.{plant['planted_ts'].replace(':', '-')}.bak")
    shutil.copyfile(mirror_path, backup)
    store.purge(ids)
    left = [r.id for r in store.all_records() if r.id in set(ids)]
    if left:
        raise PlantError(f"still present after purge: {', '.join(left)}; {plant_file} kept")
    plant_file.unlink()
    return ids


def _data_dir() -> Path:
    # The directory the UI's store and read gate use (app/ui.py, eval/prewarm.py).
    return Path(os.getenv("GATE_DIR", "data"))


def _store():
    from memory.store import default_experience_store
    return default_experience_store(_data_dir())


def _gate():
    from memory.gate import default_experience_gate
    return default_experience_gate(_data_dir())


def _test_entries() -> list[dict]:
    return load(corpus_dir(), "test")


def main(argv: list[str]) -> int:
    usage = "usage: python -m eval.demo_plant candidates | plant --item <id> | purge [--yes]"
    try:
        if argv == ["candidates"]:
            for c in candidates(_test_entries(), load_screen_rows()):
                print(f"{c['id']}\t{c['target_question']}")
        elif len(argv) == 3 and argv[:2] == ["plant", "--item"]:
            print(json.dumps(plant(argv[2], _test_entries(), _store(), _gate(), plant_file=PLANT_FILE)))
        elif argv in (["purge"], ["purge", "--yes"]):
            store = _store()
            ids = purge(store, store.jsonl_path, plant_file=PLANT_FILE, yes=argv == ["purge", "--yes"])
            print(("purged: " if argv[-1] == "--yes" else "would purge: ") + ", ".join(ids))
        else:
            print(usage, file=sys.stderr)
            return 2
    except (PlantError, FreezeError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
