import json
import subprocess
from pathlib import Path

from eval import attack_screen
from eval.attack_screen import clean_experience_entries, row_from, screen
from gate.audit import AuditLog
from gate.cache import VerdictCache
from gate.policy import DEFAULT_POLICY
from gate.read import ReadGate, ReadResult
from gate.score import aggregate
from gate.types import SignalResult, Verdict
from tests.eval.corpus_fixture import make_items


def verdict(*results, reason=None):
    v = DEFAULT_POLICY.decide(aggregate(list(results)), "corpus")
    return Verdict(v.action, v.score, reason or v.reason)


class FakeGate:
    def __init__(self):
        self.calls = []

    def screen_batch(self, items, store):
        self.calls.append((items, store))
        out = [DEFAULT_POLICY.decide(aggregate([SignalResult("structural", 0.0, "r")]), store)
               for _ in items]
        return ReadResult(out, False)


def entries():
    # interleaved across classes
    items = make_items(per_class=3, class2_groups=3)
    return [items[i] for i in (0, 3, 6, 1, 4, 7, 2, 5, 8)]


def test_corpus_classes_screened_as_corpus_and_class3_as_experience():
    g = FakeGate()
    screen(entries(), g)
    stores = {store: [i.id for i in items] for items, store in g.calls}
    assert stores["corpus"] == ["c1-0", "c2-0", "c1-1", "c2-1", "c1-2", "c2-2"]
    assert stores["experience"] == ["c3-0", "c3-1", "c3-2"]


def test_one_screen_batch_call_per_store():
    g = FakeGate()
    screen(entries(), g)
    assert sorted(s for _, s in g.calls) == ["corpus", "experience"]


def test_rows_align_with_entries():
    es = entries()
    rows = screen(es, FakeGate())
    assert [r["id"] for r in rows] == [e["id"] for e in es]


def test_row_carries_class_group_store_action_trust():
    e = next(x for x in make_items(3, 3) if x["class"] == 2)
    v = verdict(SignalResult("structural", 0.9, "x"))
    row = row_from(e, "corpus", v)
    assert row["class"] == 2 and row["group"] == e["group"] and row["store"] == "corpus"
    assert row["action"] == v.action.value
    assert row["trust"] == v.score.trust


def test_row_signals_come_from_the_verdict():
    v = verdict(SignalResult("structural", 0.0, "a"), SignalResult("corroborate", 0.9, "contradicted"))
    row = row_from(make_items(3, 3)[0], "corpus", v)
    assert row["signals"] == [{"signal": "structural", "score": 0.0, "reason": "a"},
                              {"signal": "corroborate", "score": 0.9, "reason": "contradicted"}]


def test_top_signal_is_highest_scoring():
    v = verdict(SignalResult("structural", 0.1, "a"), SignalResult("judge", 0.5, "b"))
    assert row_from(make_items(3, 3)[0], "corpus", v)["top_signal"] == "judge"
    assert row_from(make_items(3, 3)[0], "corpus", verdict())["top_signal"] is None


def test_reduced_assurance_rows_flagged():
    e = make_items(3, 3)[0]
    ok = verdict(SignalResult("structural", 0.0, "a"))
    bad = verdict(SignalResult("structural", 0.0, "a"),
                  reason="trust 1.00 clears thresholds; reduced assurance: judge unavailable")
    assert row_from(e, "corpus", ok)["reduced_assurance"] is False
    assert row_from(e, "corpus", bad)["reduced_assurance"] is True


def test_clean_experience_entries_are_class_clean(tmp_path):
    p = tmp_path / "exp.jsonl"
    rec = {"id": "r1", "task": "t", "approach": "a", "outcome": "o", "success": True, "ts": "x"}
    p.write_text(json.dumps(rec) + "\n\n", encoding="utf-8")
    [e] = clean_experience_entries(p)
    assert e == {"id": "r1", "task": "t", "approach": "a", "outcome": "o",
                 "class": "clean", "group": "r1"}


def test_empty_trusted_set_refuses_to_run(tmp_path, capsys):
    class Empty:
        def __len__(self):
            return 0

    def boom(*a, **k):
        raise AssertionError("gate built")

    rc = attack_screen.main(["--split", "test"], reference=Empty(), gate_builder=boom,
                            data_dir=tmp_path, out_root=tmp_path / "out")
    assert rc != 0
    assert "Task 43" in capsys.readouterr().err


def test_output_records_split_model_and_policy_commit(tmp_path):
    from app import config
    attack_screen.write_results(tmp_path, "test", [{"id": "x"}], policy_commit="abc123")
    out = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert out == {"split": "test", "model": config.JUDGE_MODEL,
                   "policy_commit": "abc123", "rows": [{"id": "x"}]}


def test_policy_commit_marks_a_dirty_policy_file():
    calls = []

    def git(args, cwd):
        calls.append((args[0], cwd))
        return "deadbeef"+chr(10) if args[0] == "log" else " M gate/policy.py"+chr(10)

    assert attack_screen.policy_commit(git) == "deadbeef-dirty"
    assert {c for c, _ in calls} == {"log", "status"}
    assert all(cwd == Path(attack_screen.__file__).resolve().parent.parent for _, cwd in calls)


def test_policy_commit_is_plain_when_the_policy_file_is_clean():
    git = lambda args, cwd: "deadbeef"+chr(10) if args[0] == "log" else ""
    assert attack_screen.policy_commit(git) == "deadbeef"


def test_main_takes_the_policy_commit_before_screening(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(attack_screen, "policy_commit", lambda: order.append("commit") or "c")
    monkeypatch.setattr(attack_screen, "screen", lambda entries, gate: order.append("screen") or [])
    monkeypatch.setattr(attack_screen, "corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(attack_screen, "load", lambda d, split: [])

    class Ref:
        def __len__(self):
            return 1

    rc = attack_screen.main(["--split", "test"], reference=Ref(), gate_builder=lambda *a: None,
                            data_dir=tmp_path, out_root=tmp_path / "out")
    assert rc == 0 and order == ["commit", "screen"]
    out = json.loads((tmp_path / "out" / "test" / "results.json").read_text(encoding="utf-8"))
    assert out["policy_commit"] == "c"


class CountingSignal:
    name = "structural"

    def __init__(self):
        self.calls = 0

    def score(self, item, ctx):
        self.calls += 1
        return SignalResult("structural", 0.0, "r")


def test_rerun_reuses_cached_verdicts(tmp_path):
    sig = CountingSignal()
    gate = ReadGate([sig], DEFAULT_POLICY, VerdictCache(tmp_path / "c.jsonl"),
                    AuditLog(tmp_path / "a.jsonl"))
    es = entries()
    first = [r["action"] for r in screen(es, gate)]
    n = sig.calls
    assert n > 0
    assert [r["action"] for r in screen(es, gate)] == first
    assert sig.calls == n


class OneRef:
    def __len__(self):
        return 1


def run_main(tmp_path, monkeypatch, split, seen=None, differences=()):
    monkeypatch.setattr(attack_screen, "ADAPTIVE_DIR", tmp_path / "phase5_screen")
    monkeypatch.setattr(attack_screen, "corpus_dir", lambda: tmp_path)
    monkeypatch.setattr(attack_screen, "policy_commit", lambda: "c")
    monkeypatch.setattr(attack_screen, "check", lambda path, data_dir: list(differences))

    def fake_load(d, split):
        if seen is not None:
            seen["split"] = split
        return []

    monkeypatch.setattr(attack_screen, "load", fake_load)
    monkeypatch.setattr(attack_screen, "clean_experience_entries", lambda p: [])
    argv = ["--clean-experience"] if split == "clean_experience" else ["--split", split]
    return attack_screen.main(argv, reference=OneRef(), gate_builder=lambda *a: FakeGate(),
                              data_dir=tmp_path, out_root=tmp_path / "out")


def test_adaptive_split_writes_to_phase5_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(attack_screen, "require_frozen", lambda: None)
    assert run_main(tmp_path, monkeypatch, "adaptive") == 0
    assert (tmp_path / "phase5_screen" / "results.json").exists()
    assert not (tmp_path / "out" / "adaptive").exists()


def test_adaptive_split_refuses_on_freeze_difference_before_building_gate(tmp_path, monkeypatch, capsys):
    def boom():
        raise attack_screen.FreezeError("gate differs")

    built = []
    monkeypatch.setattr(attack_screen, "require_frozen", boom)
    monkeypatch.setattr(attack_screen, "screen", lambda e, g: built.append(1) or [])
    assert run_main(tmp_path, monkeypatch, "adaptive") != 0
    assert built == [] and "gate differs" in capsys.readouterr().err
    assert not (tmp_path / "phase5_screen" / "results.json").exists()


def test_adaptive_split_loads_the_adaptive_split(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(attack_screen, "require_frozen", lambda: None)
    run_main(tmp_path, monkeypatch, "adaptive", seen)
    assert seen["split"] == "adaptive"


import pytest


@pytest.mark.parametrize("split", ["calibration", "test", "adaptive", "clean_experience"])
def test_results_record_freeze_check(tmp_path, monkeypatch, split):
    monkeypatch.setattr(attack_screen, "require_frozen", lambda: None)
    monkeypatch.setattr(attack_screen, "FROZEN_GATE", tmp_path / "frozen.json")
    (tmp_path / "frozen.json").write_text("{}", encoding="utf-8")
    run_main(tmp_path, monkeypatch, split, differences=["knob X: 1 -> 2"])
    where = tmp_path / "phase5_screen" if split == "adaptive" else tmp_path / "out" / split
    out = json.loads((where / "results.json").read_text(encoding="utf-8"))
    assert out["freeze"] == {"record": (tmp_path / "frozen.json").as_posix(),
                             "differences": ["knob X: 1 -> 2"]}


def test_test_split_runs_without_freeze_record_and_records_it(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("test split must not require a freeze")

    monkeypatch.setattr(attack_screen, "require_frozen", boom)
    monkeypatch.setattr(attack_screen, "FROZEN_GATE", tmp_path / "missing.json")
    missing = [f"no freeze record at {tmp_path / 'missing.json'}"]
    assert run_main(tmp_path, monkeypatch, "test", differences=missing) == 0
    out = json.loads((tmp_path / "out" / "test" / "results.json").read_text(encoding="utf-8"))
    assert out["freeze"] == {"record": None, "differences": missing}
