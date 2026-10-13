import json
import subprocess

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
    attack_screen.write_results(tmp_path, "test", [{"id": "x"}])
    out = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    sha = subprocess.run(["git", "log", "-1", "--format=%H", "--", "gate/policy.py"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == {"split": "test", "model": config.JUDGE_MODEL,
                   "policy_commit": sha, "rows": [{"id": "x"}]}


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
