from memory.types import ExperienceRecord
from app.orchestrator import run_agentic, format_experiences


class RecordingStore:
    def __init__(self, to_return=None):
        self.to_return = to_return or []
        self.written = []

    def retrieve(self, task, k=3):
        return self.to_return

    def write(self, record, gate=None):
        self.written.append(record)
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
