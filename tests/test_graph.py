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
                      "grade": "", "retries": 0, "web_tried": False, "answer": "",
                      "citations": [], "path_trace": []})
    assert r.calls == 1
    assert out["answer"] == "A"
    assert "rewrite" not in " ".join(out["path_trace"])

def test_weak_triggers_one_retry():
    r = FakeRetriever()
    app = build_graph(r, make_llm(["weak", "sufficient"]))
    out = app.invoke({"query": "q", "original_query": "q", "docs": [],
                      "grade": "", "retries": 0, "web_tried": False, "answer": "",
                      "citations": [], "path_trace": []})
    assert r.calls == 2
    assert any("rewrite" in s for s in out["path_trace"])
    assert out["retries"] == 1

def make_llm_seq(grade_seq):
    seq = list(grade_seq)
    import types
    m = types.SimpleNamespace()
    m.grade = lambda q, docs, client=None: seq.pop(0)
    m.rewrite = lambda q, client=None: q + " rewritten"
    m.answer = lambda q, docs, client=None: {
        "answer": "A", "citations": [f"[web: {docs[0]['paper_id']}]"] if docs and docs[0].get("source") == "web" else ["[T, chunk#0]"]}
    return m

class EmptyRetriever:
    def retrieve(self, query, k=5): return []

def make_web(docs):
    import types
    w = types.SimpleNamespace()
    w.web_search = lambda q, k=3: [dict(d) for d in docs]
    return w

def _init():
    return {"query": "q", "original_query": "q", "docs": [], "grade": "",
            "retries": 0, "web_tried": False, "answer": "", "citations": [],
            "path_trace": []}

def test_web_fallback_then_generate_when_web_sufficient():
    web_docs = [{"title": "", "paper_id": "owasp.org", "chunk_index": 0,
                 "text": "ctx", "source": "web"}]
    app = build_graph(EmptyRetriever(),
                      make_llm_seq(["sufficient"]),  # corpus weak (short-circuit), web sufficient
                      make_web(web_docs))
    out = app.invoke(_init())
    assert "web_fallback" in out["path_trace"]
    assert "generate" in out["path_trace"]
    assert out["citations"] == ["[web: owasp.org]"]
    assert out["web_tried"] is True

def test_web_weak_ends_in_refuse():
    web_docs = [{"title": "", "paper_id": "x.com", "chunk_index": 0,
                 "text": "junk", "source": "web"}]
    app = build_graph(EmptyRetriever(),
                      make_llm_seq(["weak", "weak"]),  # corpus weak, web weak
                      make_web(web_docs))
    out = app.invoke(_init())
    assert out["path_trace"][-1] == "refuse"
    assert out["answer"] == "Not grounded in the corpus."
    # web tried exactly once (no loop)
    assert out["path_trace"].count("web_fallback") == 1

def test_web_disabled_refuses(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "WEB_ENABLED", False)
    app = build_graph(EmptyRetriever(), make_llm_seq(["weak"]), make_web([]))
    out = app.invoke(_init())
    assert out["path_trace"][-1] == "refuse"
    assert "web_fallback" not in out["path_trace"]
