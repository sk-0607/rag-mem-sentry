import types
from app.graph import build_graph

class FakeRetriever:
    def __init__(self):
        self.calls = 0
    def retrieve(self, query, k=5):
        self.calls += 1
        return [{"title": "T", "chunk_index": 0, "text": "ctx", "score": 1.0}]

def make_llm(grade_seq):
    seq = list(grade_seq)
    mod = types.SimpleNamespace()
    mod.grade = lambda q, docs, client=None: seq.pop(0)
    mod.rewrite = lambda q, client=None: q + " rewritten"
    mod.answer = lambda q, docs, client=None: {"answer": "A", "citations": ["[T, chunk#0]"]}
    return mod

def test_sufficient_skips_retry():
    r = FakeRetriever()
    app = build_graph(r, make_llm(["sufficient"]))
    out = app.invoke({"query": "q", "original_query": "q", "docs": [],
                      "grade": "", "retries": 0, "answer": "",
                      "citations": [], "path_trace": []})
    assert r.calls == 1
    assert out["answer"] == "A"
    assert "rewrite" not in " ".join(out["path_trace"])

def test_weak_triggers_one_retry():
    r = FakeRetriever()
    app = build_graph(r, make_llm(["weak", "sufficient"]))
    out = app.invoke({"query": "q", "original_query": "q", "docs": [],
                      "grade": "", "retries": 0, "answer": "",
                      "citations": [], "path_trace": []})
    assert r.calls == 2
    assert any("rewrite" in s for s in out["path_trace"])
    assert out["retries"] == 1
