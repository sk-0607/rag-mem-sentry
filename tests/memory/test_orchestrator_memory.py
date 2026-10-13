import types

from app import config
from gate.score import aggregate
from gate.types import Action, Verdict
from gate.read import ReadResult
from memory.types import ExperienceRecord
from app.orchestrator import run_agentic, format_experiences


class FakeReadGate:
    """Records each (items, store) call; QUARANTINEs items whose text is in
    `exclude`, ADMITs the rest. Verdict.score uses a real (empty) aggregate
    since callers only inspect .action / .reduced_assurance here."""
    def __init__(self, exclude=frozenset(), reduced=False):
        self.exclude = exclude
        self.reduced = reduced
        self.calls = []
    def screen_batch(self, items, store):
        self.calls.append((items, store))
        verdicts = [
            Verdict(
                action=Action.QUARANTINE if item.text in self.exclude else Action.ADMIT,
                score=aggregate([]),
                reason="fake",
            )
            for item in items
        ]
        return ReadResult(verdicts=verdicts, reduced_assurance=self.reduced)


class FakeApp:
    """Stand-in for a compiled graph, for tests that only care what the
    orchestrator does before/around app.invoke, not the real graph."""
    def invoke(self, state):
        return {**state, "docs": [], "answer": "A", "citations": [],
                "path_trace": ["retrieve", "generate"]}


class RecordingStore:
    def __init__(self, to_return=None):
        self.to_return = to_return or []
        self.written = []
        self.gates = []

    def retrieve(self, task, k=3):
        return self.to_return

    def write(self, record, gate=None):
        self.written.append(record)
        self.gates.append(gate)
        return None


def rec(rid, approach):
    return ExperienceRecord(id=rid, task="t", approach=approach, outcome="o",
                            success=True, ts="2026-08-19T00:00:00Z")


def test_format_experiences_of_empty_list_is_empty_string():
    assert format_experiences([]) == ""


def test_format_experiences_includes_approach_text():
    assert "APPROACHY" in format_experiences([rec("r1", "APPROACHY")])


def test_run_without_store_reports_no_experiences(stub_llm, stub_retriever):
    out = run_agentic("what is prompt injection?", retriever=stub_retriever,
                      llm_module=stub_llm)
    assert out["experiences"] == []


def test_run_with_store_injects_retrieved_ids(stub_llm, stub_retriever):
    store = RecordingStore([rec("r1", "a")])
    out = run_agentic("what is prompt injection?", retriever=stub_retriever,
                      llm_module=stub_llm, store=store)
    assert out["experiences"] == ["r1"]


def test_run_with_store_writes_one_record(stub_llm, stub_retriever):
    store = RecordingStore()
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=stub_llm, store=store)
    assert len(store.written) == 1


def test_written_record_task_is_the_question(stub_llm, stub_retriever):
    store = RecordingStore()
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=stub_llm, store=store)
    assert store.written[0].task == "what is prompt injection?"


def test_return_shape_is_otherwise_unchanged(stub_llm, stub_retriever):
    out = run_agentic("what is prompt injection?", retriever=stub_retriever,
                      llm_module=stub_llm, store=RecordingStore())
    assert {"answer", "citations", "path_trace", "contexts"} <= set(out)


def test_an_injected_gate_is_the_one_passed_to_write(stub_llm, stub_retriever):
    sentinel = object()
    store = RecordingStore()
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=stub_llm, store=store, gate=sentinel)
    assert store.gates == [sentinel]


def test_default_gate_follows_config_not_the_process_cwd(
        stub_llm, stub_retriever, monkeypatch, tmp_path):
    # Same defect Task 19 fixed in ingest/run_ingest.py: a hardcoded Path("data")
    # writes audit/cache/quarantine wherever the process happens to be started.
    monkeypatch.setattr(config, "EXPERIENCES_PATH", str(tmp_path / "experiences.jsonl"))
    store = RecordingStore()
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=stub_llm, store=store)
    assert store.gates[0].audit.path == tmp_path / "gate_audit.jsonl"


def test_no_read_gate_passes_every_retrieved_record(stub_llm, stub_retriever):
    store = RecordingStore([rec("r1", "a")])
    out = run_agentic("what is prompt injection?", retriever=stub_retriever,
                      llm_module=stub_llm, store=store)
    assert out["experiences"] == ["r1"]
    assert out["experiences_excluded"] == []


def test_read_gate_screens_experiences_with_store_experience(stub_llm, stub_retriever):
    store = RecordingStore([rec("r1", "a")])
    gate = FakeReadGate()
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=stub_llm, store=store, read_gate=gate)
    items, store_name = gate.calls[0]
    assert store_name == "experience"
    assert items[0].id == "r1"


def test_excluded_experience_never_reaches_prompt(stub_retriever):
    record = rec("r1", "SECRETAPPROACH")
    captured = {}
    llm = types.SimpleNamespace()
    llm.decompose = lambda q, client=None: [q]
    def fake_answer(q, docs, experiences="", client=None):
        captured["experiences"] = experiences
        return {"answer": "A", "citations": []}
    llm.answer = fake_answer
    llm.synthesize = lambda q, sr, client=None: {"answer": sr[0]["answer"],
                                                  "citations": sr[0]["citations"]}
    store = RecordingStore([record])
    gate = FakeReadGate(exclude={record.to_item().text})
    run_agentic("what is prompt injection?", retriever=stub_retriever,
                llm_module=llm, store=store, read_gate=gate)
    assert "SECRETAPPROACH" not in captured["experiences"]


def test_result_splits_injected_and_excluded_ids(stub_llm, stub_retriever):
    kept = rec("r1", "keep")
    excluded = rec("r2", "drop")
    store = RecordingStore([kept, excluded])
    gate = FakeReadGate(exclude={excluded.to_item().text})
    out = run_agentic("what is prompt injection?", retriever=stub_retriever,
                      llm_module=stub_llm, store=store, read_gate=gate)
    assert out["experiences"] == ["r1"]
    assert out["experiences_excluded"] == ["r2"]


def test_read_gate_forwarded_to_graph_builder(stub_llm):
    seen = {}
    def builder(retriever, llm_module, read_gate=None):
        seen["read_gate"] = read_gate
        return FakeApp()
    sentinel = object()
    run_agentic("what is prompt injection?", retriever=object(), llm_module=stub_llm,
                graph_builder=builder, read_gate=sentinel)
    assert seen["read_gate"] is sentinel
