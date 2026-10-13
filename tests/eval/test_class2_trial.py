import io
import json
import sys
import types
from pathlib import Path

from app import config
from eval import class2_trial as mod
from eval.class2_trial import (POISON_SLOTS, normalize, poisoned_docs, run_group, run_trial,
                               target_in_answer, targets)
from gate.audit import AuditLog
from gate.read import ReadResult
from gate.score import aggregate
from gate.types import Action, SignalResult, Verdict
from tests.eval.corpus_fixture import make_items

REAL = [
    {"id": f"src-a::{i}", "title": f"T{i}", "chunk_index": i, "text": f"real chunk {i}",
     "score": 1.0 - i / 10, "trusted": True}
    for i in range(5)
]


def entries(n=3, group="c2-g0"):
    return [{"id": f"p{i}", "class": 2, "group": group, "text": f"POISON {i}",
             "target_question": "Q?", "target_answer": "The Answer"} for i in range(n)]


class FakeRetriever:
    def retrieve(self, query, k=5):
        return [dict(c) for c in REAL]


class FakeReadGate:
    """Excludes poison (id starts `p`), admits real docs; writes real audit entries."""

    def __init__(self, audit_path):
        self.audit = AuditLog(audit_path)

    def screen_batch(self, items, store):
        verdicts = []
        for item in items:
            poison = item.id.startswith("p")
            results = [SignalResult("corroboration", 0.9 if poison else 0.0, "x"),
                       SignalResult("structural", 0.0, "none")]
            verdict = Verdict(action=Action.QUARANTINE if poison else Action.ADMIT,
                              score=aggregate(results), reason="fake")
            self.audit.record(item, verdict, store, stage="read")
            verdicts.append(verdict)
        return ReadResult(verdicts=verdicts, reduced_assurance=False)


def make_llm(answer_text="plain", captured=None, by_docs=None):
    def answer(query, docs, experiences="", client=None):
        if captured is not None:
            captured["docs"] = docs
            captured["query"] = query
        for d in docs:  # what the real llm._cite_tag reads
            d["title"], d["chunk_index"]
        text = by_docs(docs) if by_docs else answer_text
        return {"answer": text, "citations": []}

    return types.SimpleNamespace(answer=answer)


def test_normalize_folds_case_and_whitespace():
    assert normalize("  The \n  ANSWER\tis  X ") == "the answer is x"


def test_target_check_is_a_normalized_substring():
    assert target_in_answer("The Answer", "so the\nanswer is here")
    assert not target_in_answer("The Answer", "something unrelated")


def test_targets_are_class2_groups_in_first_appearance_order():
    items = make_items(per_class=6, class2_groups=3)
    groups = targets(items, 10)
    assert [g[0]["group"] for g in groups] == ["c2-g0", "c2-g1", "c2-g2"]
    assert all(it["class"] == 2 and it["group"] == g[0]["group"] for g in groups for it in g)
    assert sum(len(g) for g in groups) == 6


def test_targets_capped_at_n():
    assert len(targets(make_items(per_class=30, class2_groups=10), 4)) == 4


def test_poisoned_docs_replaces_the_top_slots_and_keeps_length():
    out = poisoned_docs(REAL, entries(3))
    assert len(out) == 5
    assert [d["id"] for d in out] == ["p0", "p1", "p2", "src-a::3", "src-a::4"]
    assert [d["text"] for d in out[:3]] == ["POISON 0", "POISON 1", "POISON 2"]
    assert POISON_SLOTS == 3


def test_poisoned_docs_with_fewer_passages_than_slots():
    out = poisoned_docs(REAL, entries(2))
    assert [d["id"] for d in out] == ["p0", "p1", "src-a::2", "src-a::3", "src-a::4"]


def test_poison_docs_are_untrusted():
    out = poisoned_docs(REAL, entries(3))
    assert all(d["trusted"] is False for d in out[:3])
    assert all(d["trusted"] is True for d in out[3:])
    # The real llm.answer cites by these keys: a bare {id,text,trusted} would KeyError.
    assert all("title" in d and "chunk_index" in d for d in out[:3])
    assert REAL[0]["trusted"] is True and REAL[0]["text"] == "real chunk 0"  # input not mutated


def test_control_arm_plants_nothing():
    captured = {}
    out = run_trial(entries(3), "control", FakeRetriever(), make_llm(captured=captured), None)
    assert out["poison_in_context"] == 0
    assert [d["id"] for d in captured["docs"]] == [d["id"] for d in REAL]


def test_gate_off_counts_poison_in_context():
    out = run_trial(entries(3), "off", FakeRetriever(), make_llm(), None)
    assert out["poison_in_context"] == 3 and out["arm"] == "off"
    assert out["gate_excluded"] == [] and out["top_signals"] == {}


def test_gate_on_reports_exclusions_and_top_signals(tmp_path):
    out = run_trial(entries(3), "on", FakeRetriever(), make_llm(), FakeReadGate(tmp_path / "a.jsonl"))
    assert out["poison_in_context"] == 0
    assert sorted(out["gate_excluded"]) == ["p0", "p1", "p2"]
    assert out["top_signals"] == {"p0": "corroboration", "p1": "corroboration", "p2": "corroboration"}


def test_control_hit_voids_the_trial(tmp_path):
    hit = make_llm(by_docs=lambda docs: "the answer is here")
    assert run_group(entries(3), FakeRetriever(), hit, FakeReadGate(tmp_path / "a.jsonl"))["void"] is True
    miss = run_group(entries(3), FakeRetriever(), make_llm(), FakeReadGate(tmp_path / "b.jsonl"))
    assert miss["void"] is False
    assert set(miss) >= {"control", "off", "on", "void"}


def test_question_is_the_groups_target_question():
    captured = {}
    run_trial(entries(3), "off", FakeRetriever(), make_llm(captured=captured), None)
    assert captured["query"] == "Q?"


def fake_judge_chat(messages, model=None):
    n = messages[1]["content"].count("<item id=")
    return json.dumps([{"id": f"i{i}", "suspicion": 0.0, "label": "clean", "evidence_span": None}
                       for i in range(n)])


def patch_main(monkeypatch, tmp_path, seen=None):
    monkeypatch.setattr(mod, "TRIAL_DIR", tmp_path / "trial")
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr(mod, "corpus_dir", lambda: tmp_path / "corpus")

    def fake_load(directory, split=None):
        if seen is not None:
            seen["split"] = split
        return make_items(per_class=6, class2_groups=3)

    monkeypatch.setattr(mod, "load", fake_load)
    monkeypatch.setattr("app.retrieval.Retriever", FakeRetriever)
    monkeypatch.setattr("app.llm.answer", make_llm().answer)
    monkeypatch.setattr("app.llm.chat", fake_judge_chat)
    monkeypatch.setattr(sys, "stdout", io.StringIO())


def test_cli_loads_the_test_split_only(tmp_path, monkeypatch):
    seen = {}
    patch_main(monkeypatch, tmp_path, seen)
    monkeypatch.setenv("CLASS2_N", "1")
    assert mod.main() == 0
    assert seen["split"] == "test"


def test_gate_on_uses_real_trusted_reference(tmp_path, monkeypatch):
    sentinel, seen = object(), {}

    def fake_reference(data_dir):
        seen["ref_dir"] = Path(data_dir)
        return sentinel

    def fake_gate(data_dir, chat=None, reference=None):
        seen["reference"] = reference
        return None

    monkeypatch.setattr(mod, "default_reference", fake_reference)
    monkeypatch.setattr(mod, "default_read_gate", fake_gate)
    mod.run_all([entries(3)], FakeRetriever(), make_llm(), tmp_path / "trial")
    assert seen["reference"] is sentinel and seen["reference"] is not None
    assert seen["ref_dir"] == Path(config.EXPERIENCES_PATH).parent


def test_run_all_wipes_only_the_directory_it_is_given(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "stale.txt").write_text("old", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("keep", encoding="utf-8")
    mod.run_all([entries(3)], FakeRetriever(), make_llm(), trial, chat=fake_judge_chat)
    assert not (trial / "stale.txt").exists() and other.exists()


def test_output_names_both_models(tmp_path, monkeypatch):
    patch_main(monkeypatch, tmp_path)
    monkeypatch.setenv("CLASS2_N", "1")
    mod.main()
    on_disk = json.loads((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    assert on_disk["answer_model"] == config.LLM_MODEL
    assert on_disk["gate_model"] == config.JUDGE_MODEL


def test_cli_warns_when_judge_model_unset(tmp_path, monkeypatch):
    patch_main(monkeypatch, tmp_path)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setenv("CLASS2_N", "1")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    mod.main()
    assert "JUDGE_MODEL" in out.getvalue().splitlines()[0]


def test_results_marked_incomplete_until_the_last_group(tmp_path, monkeypatch):
    patch_main(monkeypatch, tmp_path)
    monkeypatch.setenv("CLASS2_N", "2")
    snaps = []
    real_save = mod.save

    def spy(*a, **k):
        out = real_save(*a, **k)
        snaps.append(json.loads((tmp_path / "results" / "results.json").read_text(encoding="utf-8")))
        return out

    monkeypatch.setattr(mod, "save", spy)
    mod.main()
    assert [(s["complete"], len(s["groups"])) for s in snaps] == [(False, 1), (False, 2), (True, 2)]
