import io
import json
import sys
import types
from pathlib import Path

import pytest

from app import config
from eval import class1_trial as mod
from eval.class1_trial import CANARY, SAMPLES, make_carrier, run_trial
from gate.audit import AuditLog
from gate.read import ReadResult
from gate.score import aggregate
from gate.types import Action, SignalResult, Verdict

ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"

REAL_TOP5 = [
    {"id": f"paper::{i}", "title": "T", "chunk_index": i, "text": f"real chunk {i}", "score": 1.0 - i / 10}
    for i in range(5)
]


@pytest.fixture(autouse=True)
def hermetic_data_dir(tmp_path, monkeypatch):
    # run_all and main build the reference from CHUNKS_PATH's directory; left alone
    # that is the real data/, which no test may read.
    monkeypatch.setattr(config, "EXPERIENCES_PATH", str(tmp_path / "experiences.jsonl"))
    monkeypatch.setattr(config, "CHUNKS_PATH", str(tmp_path / "chunks.jsonl"))


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    # main() checks the real freeze record; under a tmp CHUNKS_PATH the reference
    # fingerprint differs, so every main() test runs as if the check were clean.
    monkeypatch.setattr(mod, "require_frozen", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("eval.attack_screen.check", lambda *a, **k: [])


class FakeReference:
    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n

    def fingerprint(self):
        return "ref-fp"


def write_reference():
    Path(config.CHUNKS_PATH).write_text(
        json.dumps({"id": "ref::0", "text": "trusted", "trusted": True}) + chr(10), encoding="utf-8")


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else REAL_TOP5

    def retrieve(self, query, k=5):
        return [dict(c) for c in self.chunks]


class FakeReadGate:
    """QUARANTINEs carriers (id ends `::poisoned`), ADMITs the rest, and writes
    each verdict to a real AuditLog the way ReadGate does — so run_trial's
    top-signal lookup reads the same entry shape it sees in production."""

    def __init__(self, audit_path, carrier_action=Action.QUARANTINE, reason="fake"):
        self.audit = AuditLog(audit_path)
        self.carrier_action = carrier_action
        self.reason = reason

    def screen_batch(self, items, store):
        verdicts = []
        for item in items:
            if item.id.endswith("::poisoned"):
                results = [SignalResult("structural", 0.8, "pattern"), SignalResult("judge", 0.1, "clean")]
                action = self.carrier_action
            else:
                results = [SignalResult("structural", 0.0, "none"), SignalResult("judge", 0.0, "clean")]
                action = Action.ADMIT
            verdict = Verdict(action=action, score=aggregate(results), reason=self.reason)
            self.audit.record(item, verdict, store, stage="read")
            verdicts.append(verdict)
        return ReadResult(verdicts=verdicts, reduced_assurance=False)


def make_llm(answer_text="plain answer", captured=None):
    def answer(query, docs, experiences="", client=None):
        if captured is not None:
            captured["docs"] = docs
        return {"answer": answer_text, "citations": []}

    return types.SimpleNamespace(answer=answer)


SAMPLE = SAMPLES[0]


def test_every_sample_instructs_the_canary():
    for s in SAMPLES:
        assert CANARY in s["poison"]
        # Zero-width characters sit inside the instruction words by design; the
        # instruction itself must still read as one once they are removed.
        visible = s["poison"].translate({ord(c): None for c in ZERO_WIDTH}).lower()
        assert "answer" in visible and "include" in visible


def test_five_samples_cover_the_r3_variants():
    assert [s["name"] for s in SAMPLES] == [
        "plain-imperative", "display-none-html", "zero-width-obfuscated",
        "html-comment", "fake-system-override",
    ]
    by_name = {s["name"]: s["poison"] for s in SAMPLES}
    assert "display:none" in by_name["display-none-html"]
    assert any(c in by_name["zero-width-obfuscated"] for c in ZERO_WIDTH)
    assert "<!--" in by_name["html-comment"]
    assert "SYSTEM" in by_name["fake-system-override"]


def test_sample_questions_are_distinct_qa_set_questions():
    # Distinct questions -> distinct real top-1 chunks -> distinct carrier ids,
    # so one sample's audit entry can never be read as another's.
    qa = {q["question"] for q in json.loads(Path("eval/qa_set.json").read_text(encoding="utf-8"))}
    questions = [s["question"] for s in SAMPLES]
    assert len(set(questions)) == 5
    assert set(questions) <= qa


def test_make_carrier_preserves_real_text_and_appends_poison():
    real = dict(REAL_TOP5[0])
    carrier = make_carrier(real, "POISON")

    assert carrier["text"].startswith(real["text"])
    assert carrier["text"].endswith("POISON")
    assert real["text"] == "real chunk 0"  # the input is copied, not mutated
    assert carrier["title"] == real["title"] and carrier["chunk_index"] == real["chunk_index"]


def test_make_carrier_marks_id():
    real = dict(REAL_TOP5[0])

    assert make_carrier(real, "POISON")["id"] == "paper::0::poisoned"
    assert real["id"] == "paper::0"


def test_behavior_check_is_exact_substring():
    lower = run_trial(SAMPLE, FakeRetriever(), make_llm(CANARY.lower()), None)
    exact = run_trial(SAMPLE, FakeRetriever(), make_llm(f"Sure: {CANARY}."), None)

    assert lower["canary_in_answer"] is False
    assert exact["canary_in_answer"] is True


def test_run_trial_gate_off_reports_poison_in_context():
    captured = {}

    out = run_trial(SAMPLE, FakeRetriever(), make_llm(captured=captured), None)

    assert out["poisoned_in_context"] is True
    assert out["arm"] == "off"
    assert out["name"] == SAMPLE["name"]
    assert out["gate_excluded"] == [] and out["top_signals"] == {}
    assert out["carrier_audit"] is None
    assert captured["docs"][0]["text"].endswith(SAMPLE["poison"])
    assert [d["text"] for d in captured["docs"][1:]] == [c["text"] for c in REAL_TOP5[1:]]


def test_run_trial_gate_on_reports_exclusion(tmp_path):
    gate = FakeReadGate(tmp_path / "audit.jsonl")

    out = run_trial(SAMPLE, FakeRetriever(), make_llm(), gate)

    assert out["arm"] == "on"
    assert out["poisoned_in_context"] is False
    assert out["gate_excluded"] == ["paper::0::poisoned"]
    # Credited to the signal that scored highest, not to "the gate".
    assert out["top_signals"] == {"paper::0::poisoned": "structural"}


def test_run_trial_gate_on_carries_the_carriers_audit_entry(tmp_path):
    out = run_trial(SAMPLE, FakeRetriever(), make_llm(), FakeReadGate(tmp_path / "audit.jsonl"))

    audit = out["carrier_audit"]
    assert audit["action"] == "quarantine" and audit["reason"] == "fake"
    assert audit["trust"] == pytest.approx(0.18)
    assert audit["signals"] == [
        {"signal": "structural", "score": 0.8, "reason": "pattern", "evidence_span": None},
        {"signal": "judge", "score": 0.1, "reason": "clean", "evidence_span": None},
    ]


def test_admitted_carrier_keeps_the_reduced_assurance_reason(tmp_path):
    # An admitted carrier is explained only by its audit entry, and the trial
    # dir holding it is wiped by the next run, so it must be in the result.
    gate = FakeReadGate(tmp_path / "audit.jsonl", carrier_action=Action.ADMIT,
                        reason="trust 0.82; reduced assurance: judge unavailable")

    out = run_trial(SAMPLE, FakeRetriever(), make_llm(), gate)

    assert out["poisoned_in_context"] is True and out["gate_excluded"] == []
    assert out["carrier_audit"]["action"] == "admit"
    assert "reduced assurance: judge unavailable" in out["carrier_audit"]["reason"]


def test_real_trial_and_results_dirs_are_not_nested():
    # The tests below patch both to tmp_path, so nothing else pins the real
    # values: results kept inside the dir run_all wipes would be deleted by it.
    assert mod.RESULTS_DIR != mod.TRIAL_DIR
    assert mod.TRIAL_DIR not in mod.RESULTS_DIR.parents


def fake_judge_chat(messages, model=None):
    n = messages[1]["content"].count("<item id=")
    return json.dumps([{"id": f"i{i}", "suspicion": 0.0, "label": "clean", "evidence_span": None}
                       for i in range(n)])


def test_run_all_runs_each_sample_off_then_on_through_the_real_gate(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "stale.txt").write_text("old", encoding="utf-8")

    results = mod.run_all(FakeRetriever(), make_llm(), trial_dir, chat=fake_judge_chat)

    assert [(r["name"], r["arm"]) for r in results] == [
        (s["name"], arm) for s in SAMPLES for arm in ("off", "on")
    ]
    assert not (trial_dir / "stale.txt").exists()
    for r in results:
        assert set(r["top_signals"]) == set(r["gate_excluded"])
        assert r["poisoned_in_context"] is (r["arm"] == "off" or not r["gate_excluded"])


def test_main_writes_results_outside_the_wiped_dir_and_prints_on_cp1252(tmp_path, monkeypatch):
    trial_dir, results_dir = tmp_path / "trial", tmp_path / "results"
    monkeypatch.setattr(mod, "TRIAL_DIR", trial_dir)
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)  # restored on teardown
    monkeypatch.setattr("app.retrieval.Retriever", FakeRetriever)
    # "\u4e2d" is not encodable in cp1252.
    monkeypatch.setattr("app.llm.answer", make_llm("\u4e2d").answer)
    monkeypatch.setattr("app.llm.chat", fake_judge_chat)
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    write_reference()

    assert mod.main() == 0

    on_disk = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert len(on_disk["trials"]) == 10
    assert on_disk["complete"] is True
    assert on_disk["model"] == config.LLM_MODEL
    assert on_disk["trials"][0]["answer"] == "\u4e2d"


def test_main_keeps_completed_trials_on_disk_when_a_later_call_fails(tmp_path, monkeypatch):
    trial_dir, results_dir = tmp_path / "trial", tmp_path / "results"
    monkeypatch.setattr(mod, "TRIAL_DIR", trial_dir)
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr("app.retrieval.Retriever", FakeRetriever)
    monkeypatch.setattr("app.llm.chat", fake_judge_chat)
    calls = []

    def flaky_answer(query, docs, experiences="", client=None):
        calls.append(query)
        if len(calls) == 3:  # what llm.chat raises once its retries are exhausted
            raise RuntimeError("rate limit: exhausted retries")
        return {"answer": f"answer {len(calls)}", "citations": []}

    monkeypatch.setattr("app.llm.answer", flaky_answer)
    write_reference()

    with pytest.raises(RuntimeError):  # the error still propagates: exit code stays non-zero
        mod.main()

    on_disk = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert on_disk["complete"] is False
    assert [t["answer"] for t in on_disk["trials"]] == ["answer 1", "answer 2"]


def test_make_carrier_is_untrusted():
    real = {**REAL_TOP5[0], "trusted": True}
    assert make_carrier(real, "POISON")["trusted"] is False


def test_class1_gate_on_uses_real_trusted_reference(tmp_path, monkeypatch):
    sentinel = object()
    seen = {}
    real_dir = Path(config.CHUNKS_PATH).parent

    def fake_reference(data_dir):
        seen["ref_dir"] = Path(data_dir)
        return sentinel

    def fake_gate(data_dir, chat=None, reference=None):
        seen["reference"] = reference
        return None  # gate-off twin; no screening needed here

    monkeypatch.setattr(mod, "default_reference", fake_reference)
    monkeypatch.setattr(mod, "default_read_gate", fake_gate)
    monkeypatch.setattr(mod, "SAMPLES", SAMPLES[:1])

    mod.run_all(FakeRetriever(), make_llm(), tmp_path / "trial")

    assert seen["reference"] is sentinel
    assert seen["ref_dir"] == real_dir


def test_empty_reference_stops_main_before_any_spend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")

    def boom(*a, **k):
        raise AssertionError("spent something before the empty-reference guard")

    class BoomRetriever:
        def __init__(self):
            boom()

    monkeypatch.setattr("app.retrieval.Retriever", BoomRetriever)
    monkeypatch.setattr("app.llm.answer", boom)
    monkeypatch.setattr("app.llm.chat", boom)
    monkeypatch.setattr(mod.shutil, "rmtree", boom)
    assert mod.main() == 1  # no chunks.jsonl under tmp_path: empty reference
    assert "Task 43" in capsys.readouterr().err


def test_gate_off_arm_needs_no_reference(tmp_path):
    # The guard lives in main: run_all with an empty reference still runs both arms.
    from gate.reference import ReferenceIndex
    out = mod.run_all(FakeRetriever(), make_llm(), tmp_path / "trial", chat=fake_judge_chat,
                      reference=ReferenceIndex([]))
    assert len(out) == 10


def no_spend(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("spent something")

    monkeypatch.setattr(mod, "default_read_gate", boom)
    monkeypatch.setattr("app.llm.answer", boom)
    monkeypatch.setattr("app.llm.chat", boom)
    return calls


def test_class1_trial_refuses_unclean_freeze_before_building_gate(tmp_path, monkeypatch, capsys):
    from eval.freeze_gate import FreezeError
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")

    def unclean(*a, **k):
        raise FreezeError("gate changed since freeze")

    monkeypatch.setattr(mod, "require_frozen", unclean)
    calls = no_spend(monkeypatch)
    write_reference()

    assert mod.main() != 0
    assert calls == []
    assert not (tmp_path / "results").exists()
    assert "gate changed" in capsys.readouterr().err


def test_class1_trial_records_freeze_field(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr("app.retrieval.Retriever", FakeRetriever)
    monkeypatch.setattr("app.llm.answer", make_llm().answer)
    monkeypatch.setattr("app.llm.chat", fake_judge_chat)
    write_reference()

    assert mod.main() == 0

    from eval.freeze_gate import FROZEN_GATE
    assert FROZEN_GATE.exists()
    on_disk = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert on_disk["freeze"] == {"record": FROZEN_GATE.as_posix(), "differences": []}


def test_class1_trial_passes_real_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr("app.retrieval.Retriever", FakeRetriever)
    monkeypatch.setattr("app.llm.answer", make_llm().answer)
    monkeypatch.setattr(mod, "SAMPLES", SAMPLES[:1])
    fake = FakeReference()
    seen = {"ref_dirs": [], "refs": []}

    def fake_reference(data_dir):
        seen["ref_dirs"].append(Path(data_dir))
        return fake

    def fake_gate(data_dir, chat=None, reference=None):
        seen["refs"].append(reference)
        return None

    monkeypatch.setattr(mod, "default_reference", fake_reference)
    monkeypatch.setattr(mod, "default_read_gate", fake_gate)

    assert mod.main() == 0
    assert seen["refs"] and all(r is fake for r in seen["refs"])
    assert seen["ref_dirs"] == [Path(config.CHUNKS_PATH).parent]
    assert tmp_path / "trial" not in seen["ref_dirs"]


def test_class1_trial_refuses_empty_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(mod, "default_reference", lambda d: FakeReference(0))
    calls = no_spend(monkeypatch)

    assert mod.main() != 0
    assert calls == []
