import json
from dataclasses import asdict
from pathlib import Path

import pytest

from eval import demo_plant as mod
from eval.attack_corpus import to_gate_item
from eval.demo_plant import PlantError, candidates, load_screen_rows, plant, purge, purge_ids
from eval.freeze_gate import FreezeError
from gate.score import aggregate
from gate.types import Action, SignalResult, Verdict
from memory.types import ExperienceRecord
from tests.eval.corpus_fixture import make_items

ITEMS = make_items()
C3 = next(e for e in ITEMS if e["class"] == 3)
C1 = next(e for e in ITEMS if e["class"] == 1)
PLANTED_TS = "2026-10-15T10:00:00+00:00"


class FakeStore:
    """JSONL mirror only, the same contract as memory.store.ExperienceStore:
    write persists on admit only, purge removes by id."""

    def __init__(self, path):
        self.jsonl_path = Path(path)

    def write(self, record, gate=None):
        verdict = gate.screen(record.to_item(), "experience") if gate is not None else None
        if verdict is None or verdict.action is Action.ADMIT:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        return verdict

    def all_records(self):
        if not self.jsonl_path.exists():
            return []
        return [ExperienceRecord(**json.loads(l))
                for l in self.jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def purge(self, ids):
        keep = [r for r in self.all_records() if r.id not in set(ids)]
        self.jsonl_path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in keep), encoding="utf-8")
        return len(self.all_records()) - len(keep)


class StickyStore(FakeStore):
    def purge(self, ids):
        super().purge(list(ids)[1:])  # leaves the first id behind


class FakeGate:
    def __init__(self, action=Action.ADMIT):
        self.action = action

    def screen(self, item, store):
        return Verdict(action=self.action, score=aggregate([SignalResult("structural", 0.0, "x")]),
                       reason=f"fake {self.action.value}")


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    monkeypatch.setattr(mod, "require_frozen", lambda *a, **k: None)


def rec(rid, ts):
    return ExperienceRecord(id=rid, task="t", approach="a", outcome="o", success=True, ts=ts)


def row(entry, action="quarantine", top="judge", reduced=False):
    return {"id": entry["id"], "class": entry["class"], "group": entry["group"], "store": "x",
            "action": action, "trust": 0.1, "top_signal": top, "signals": [],
            "reduced_assurance": reduced}


# --- candidates -------------------------------------------------------------

def test_candidates_keeps_only_excluded_class3_judge_topped_test_rows():
    c3 = [e for e in ITEMS if e["class"] == 3]
    rows = [row(c3[0]), row(c3[1], action="admit"), row(c3[2], top="structural"),
            row(C1), row(c3[3], action="block")]
    out = candidates(ITEMS, rows)
    assert [c["id"] for c in out] == [c3[0]["id"], c3[3]["id"]]
    assert out[0]["target_question"] == c3[0]["target_question"]


def test_candidates_skips_reduced_assurance_rows():
    assert candidates(ITEMS, [row(C3, reduced=True)]) == []


def test_candidates_missing_screen_file_raises_naming_path(tmp_path):
    missing = tmp_path / "nope" / "results.json"
    with pytest.raises(PlantError, match="nope"):
        load_screen_rows(missing)


# --- plant ------------------------------------------------------------------

def test_plant_refuses_existing_plant_file_and_writes_nothing(tmp_path):
    pf = tmp_path / "plant.json"
    pf.write_text("{}", encoding="utf-8")
    store = FakeStore(tmp_path / "exp.jsonl")
    with pytest.raises(PlantError):
        plant(C3["id"], ITEMS, store, FakeGate(), plant_file=pf)
    assert store.all_records() == [] and pf.read_text(encoding="utf-8") == "{}"


def test_plant_refuses_non_class3_id(tmp_path):
    store = FakeStore(tmp_path / "exp.jsonl")
    with pytest.raises(PlantError, match="class"):
        plant(C1["id"], ITEMS, store, FakeGate(), plant_file=tmp_path / "plant.json")
    assert store.all_records() == [] and not (tmp_path / "plant.json").exists()


def test_plant_refuses_id_outside_test_split(tmp_path):
    store = FakeStore(tmp_path / "exp.jsonl")
    with pytest.raises(PlantError, match="not in the test split"):
        plant("no-such-id", ITEMS, store, FakeGate(), plant_file=tmp_path / "plant.json")
    assert store.all_records() == [] and not (tmp_path / "plant.json").exists()


def test_plant_refuses_unclean_freeze(tmp_path, monkeypatch):
    def unclean(*a, **k):
        raise FreezeError("gate changed")

    monkeypatch.setattr(mod, "require_frozen", unclean)
    store = FakeStore(tmp_path / "exp.jsonl")
    with pytest.raises(FreezeError):
        plant(C3["id"], ITEMS, store, FakeGate(), plant_file=tmp_path / "plant.json")
    assert store.all_records() == [] and not (tmp_path / "plant.json").exists()


def test_plant_non_admit_persists_nothing_and_raises_with_verdict(tmp_path):
    store = FakeStore(tmp_path / "exp.jsonl")
    with pytest.raises(PlantError, match="quarantine"):
        plant(C3["id"], ITEMS, store, FakeGate(Action.QUARANTINE), plant_file=tmp_path / "plant.json")
    assert store.all_records() == [] and not (tmp_path / "plant.json").exists()


def test_plant_admit_writes_plant_file_with_four_keys(tmp_path):
    store = FakeStore(tmp_path / "exp.jsonl")
    pf = tmp_path / "plant.json"
    out = plant(C3["id"], ITEMS, store, FakeGate(), plant_file=pf, now=lambda: PLANTED_TS)
    on_disk = json.loads(pf.read_text(encoding="utf-8"))
    assert on_disk == out == {"item_id": C3["id"], "record_id": C3["id"], "planted_ts": PLANTED_TS,
                              "target_question": C3["target_question"]}
    assert [r.id for r in store.all_records()] == [C3["id"]]
    # The planted record is the corpus projection: the text the gate screened is the item's.
    assert store.all_records()[0].to_item().text == to_gate_item(C3)[0].text


def test_plant_makes_no_chat_call(tmp_path, monkeypatch):
    from memory.gate import default_experience_gate

    def boom(*a, **k):
        raise AssertionError("LLM call during plant")

    monkeypatch.setattr("app.llm.chat", boom)
    store = FakeStore(tmp_path / "exp.jsonl")
    plant(C3["id"], ITEMS, store, default_experience_gate(tmp_path / "gate"),
          plant_file=tmp_path / "plant.json")
    assert (tmp_path / "plant.json").exists()


# --- purge ------------------------------------------------------------------

def planted(tmp_path, store_cls=FakeStore):
    store = store_cls(tmp_path / "exp.jsonl")
    for r in (rec("earlier", "2026-10-14T09:00:00+00:00"),
              rec("planted", "1970-01-01T00:00:00+00:00"),
              rec("later", "2026-10-15T11:00:00+00:00")):
        store.write(r)
    pf = tmp_path / "plant.json"
    pf.write_text(json.dumps({"item_id": "planted", "record_id": "planted",
                              "planted_ts": PLANTED_TS, "target_question": "q"}), encoding="utf-8")
    return store, pf


def test_purge_ids_include_planted_and_later_records_only(tmp_path):
    store, pf = planted(tmp_path)
    ids = purge_ids(json.loads(pf.read_text(encoding="utf-8")), store.all_records())
    assert ids == ["planted", "later"]


def test_purge_without_yes_changes_nothing(tmp_path):
    store, pf = planted(tmp_path)
    before = store.jsonl_path.read_bytes()
    assert purge(store, store.jsonl_path, plant_file=pf) == ["planted", "later"]
    assert store.jsonl_path.read_bytes() == before and pf.exists()
    assert list(tmp_path.glob("*.bak")) == []


def test_purge_with_yes_backs_up_before_removing(tmp_path):
    store, pf = planted(tmp_path)
    before = store.jsonl_path.read_bytes()
    assert purge(store, store.jsonl_path, plant_file=pf, yes=True) == ["planted", "later"]
    backups = list(tmp_path.glob("exp.jsonl.*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert ":" not in backups[0].name
    assert [r.id for r in store.all_records()] == ["earlier"]
    assert not pf.exists()


def test_purge_reports_survivor_and_keeps_plant_file(tmp_path):
    store, pf = planted(tmp_path, StickyStore)
    with pytest.raises(PlantError, match="planted"):
        purge(store, store.jsonl_path, plant_file=pf, yes=True)
    assert pf.exists()


def test_purge_without_plant_file_exits_1_and_changes_nothing(tmp_path, monkeypatch, capsys):
    store = FakeStore(tmp_path / "exp.jsonl")
    store.write(rec("keep", PLANTED_TS))
    before = store.jsonl_path.read_bytes()
    with pytest.raises(PlantError, match="nothing planted"):
        purge(store, store.jsonl_path, plant_file=tmp_path / "plant.json", yes=True)
    monkeypatch.setattr(mod, "PLANT_FILE", tmp_path / "plant.json")
    monkeypatch.setattr(mod, "_store", lambda: store)
    assert mod.main(["purge", "--yes"]) == 1
    assert "nothing planted" in capsys.readouterr().err
    assert store.jsonl_path.read_bytes() == before and list(tmp_path.glob("*.bak")) == []


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    store = FakeStore(tmp_path / "exp.jsonl")
    monkeypatch.setattr(mod, "PLANT_FILE", tmp_path / "plant.json")
    monkeypatch.setattr(mod, "_store", lambda: store)
    monkeypatch.setattr(mod, "_gate", lambda: FakeGate())
    monkeypatch.setattr(mod, "_test_entries", lambda: ITEMS)
    monkeypatch.setattr(mod, "load_screen_rows", lambda path=None: [row(C3)])

    assert mod.main(["candidates"]) == 0
    assert C3["id"] in capsys.readouterr().out
    assert mod.main(["plant", "--item", C1["id"]]) == 1
    assert mod.main(["plant", "--item", C3["id"]]) == 0
    assert mod.main(["plant", "--item", C3["id"]]) == 1  # already planted
    assert mod.main(["purge"]) == 0
    assert mod.main(["purge", "--yes"]) == 0
    assert mod.main(["purge", "--yes"]) == 1  # nothing planted now

    def unclean(*a, **k):
        raise FreezeError("gate changed")

    monkeypatch.setattr(mod, "require_frozen", unclean)
    assert mod.main(["plant", "--item", C3["id"]]) == 1
