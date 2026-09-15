import types
from app.graph import build_graph

class FakeRetriever:
    def __init__(self):
        self.calls = 0
    def retrieve(self, query, k=5):
        self.calls += 1
        return [{"title": "T", "chunk_index": 0, "text": "ctx", "score": 1.0}]

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
