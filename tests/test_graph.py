import types
from app.graph import build_graph
from gate.cache import content_hash
from gate.score import aggregate
from gate.types import Action, Verdict
from gate.read import ReadResult

ONE_CHUNK = [{"title": "T", "chunk_index": 0, "text": "ctx", "score": 1.0}]
TWO_CHUNKS = [
    {"title": "T", "chunk_index": 0, "text": "keep me", "score": 1.0},
    {"title": "T", "chunk_index": 1, "text": "exclude me", "score": 0.9},
]

class FakeRetriever:
    def __init__(self, chunks=None):
        self.calls = 0
        self._chunks = chunks if chunks is not None else ONE_CHUNK
    def retrieve(self, query, k=5):
        self.calls += 1
        return self._chunks

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

def make_llm(captured=None):
    mod = types.SimpleNamespace()
    def fake_answer(q, docs, experiences="", client=None):
        if captured is not None:
            captured["experiences"] = experiences
        return {"answer": "A", "citations": ["[T, chunk#0]"]}
    mod.answer = fake_answer
    return mod

def _init(**extra):
    return {"query": "q", "docs": [], "answer": "", "citations": [],
            "path_trace": [], **extra}

def test_retrieves_once_then_generates():
    r = FakeRetriever()
    out = build_graph(r, make_llm()).invoke(_init())
    assert r.calls == 1
    assert out["answer"] == "A"
    assert out["citations"] == ["[T, chunk#0]"]
    assert out["path_trace"] == ["retrieve", "generate"]

def test_generate_forwards_experiences_to_answer():
    captured = {}
    exp = "Past experiences:\n- task: t\n  approach: a\n  outcome: o"
    out = build_graph(FakeRetriever(), make_llm(captured)).invoke(_init(experiences=exp))
    assert captured["experiences"] == exp
    assert out["answer"] == "A"


def test_generate_defaults_experiences_to_empty_when_absent_from_state():
    captured = {}
    build_graph(FakeRetriever(), make_llm(captured)).invoke(_init())   # no "experiences" key at all
    assert captured["experiences"] == ""


def test_no_read_gate_trace_unchanged():
    out = build_graph(FakeRetriever(), make_llm()).invoke(_init())
    assert out["path_trace"] == ["retrieve", "generate"]


def test_gate_node_between_retrieve_and_generate():
    gate = FakeReadGate(exclude={"exclude me"})
    out = build_graph(FakeRetriever(TWO_CHUNKS), make_llm(), read_gate=gate).invoke(_init())
    assert out["path_trace"] == ["retrieve", "gate:1/2", "generate"]


def test_excluded_chunk_never_reaches_answer():
    captured = {}
    def fake_answer(q, docs, experiences="", client=None):
        captured["docs"] = docs
        return {"answer": "A", "citations": []}
    llm = types.SimpleNamespace(answer=fake_answer)
    gate = FakeReadGate(exclude={"exclude me"})
    build_graph(FakeRetriever(TWO_CHUNKS), llm, read_gate=gate).invoke(_init())
    texts = [d["text"] for d in captured["docs"]]
    assert texts == ["keep me"]


def test_gate_excluded_lists_excluded_ids():
    gate = FakeReadGate(exclude={"exclude me"})
    out = build_graph(FakeRetriever(TWO_CHUNKS), make_llm(), read_gate=gate).invoke(_init())
    assert out["gate_excluded"] == [content_hash("exclude me")[:16]]


def test_reduced_assurance_appended_to_trace():
    gate = FakeReadGate(reduced=True)
    out = build_graph(FakeRetriever(), make_llm(), read_gate=gate).invoke(_init())
    assert out["path_trace"] == ["retrieve", "gate:1/1", "gate:reduced_assurance", "generate"]


def test_all_excluded_still_generates_with_empty_docs():
    captured = {}
    def fake_answer(q, docs, experiences="", client=None):
        captured["docs"] = docs
        return {"answer": "A", "citations": []}
    llm = types.SimpleNamespace(answer=fake_answer)
    gate = FakeReadGate(exclude={"ctx"})
    out = build_graph(FakeRetriever(), llm, read_gate=gate).invoke(_init())
    assert captured["docs"] == []
    assert out["path_trace"] == ["retrieve", "gate:0/1", "generate"]


def test_chunk_without_id_gets_content_derived_item_id():
    gate = FakeReadGate()
    build_graph(FakeRetriever(), make_llm(), read_gate=gate).invoke(_init())
    items, _ = gate.calls[0]
    assert items[0].id == content_hash("ctx")[:16]


def test_gate_screens_with_store_corpus():
    gate = FakeReadGate()
    build_graph(FakeRetriever(), make_llm(), read_gate=gate).invoke(_init())
    _, store = gate.calls[0]
    assert store == "corpus"


import pytest


@pytest.mark.parametrize("stored,expected", [({"trusted": True}, True), ({}, False), ({"trusted": "true"}, False)])
def test_gate_item_carries_trust_flag(stored, expected):
    gate = FakeReadGate()
    build_graph(FakeRetriever([{**ONE_CHUNK[0], **stored}]), make_llm(), read_gate=gate).invoke(_init())
    items, _ = gate.calls[0]
    assert items[0].metadata == {"trusted": expected}
