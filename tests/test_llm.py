from app import llm

def test_format_citations():
    docs = [{"title": "PoisonedRAG", "chunk_index": 3},
            {"title": "GCG", "chunk_index": 0}]
    assert llm.format_citations(docs) == ["[PoisonedRAG, chunk#3]", "[GCG, chunk#0]"]

def test_citations_web_vs_corpus_and_dedupe():
    docs = [
        {"title": "PoisonedRAG", "chunk_index": 3, "source": "corpus"},
        {"title": "PoisonedRAG", "chunk_index": 3, "source": "corpus"},  # dup
        {"paper_id": "arxiv.org", "source": "web"},
    ]
    cites = llm.format_citations(docs)
    assert cites == ["[PoisonedRAG, chunk#3]", "[web: arxiv.org]"]

def test_answer_refuses_when_no_docs():
    out = llm.answer("anything", [])
    assert out["answer"] == "Not grounded in the corpus."
    assert out["citations"] == []

def test_answer_uses_chat_and_returns_citations(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda messages, client=None: "Prompt injection is X.")
    docs = [{"title": "T", "chunk_index": 1, "text": "prompt injection detail"}]
    out = llm.answer("what is prompt injection?", docs)
    assert out["answer"] == "Prompt injection is X."
    assert out["citations"] == ["[T, chunk#1]"]

def test_grade_parses_weak(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda messages, client=None: "WEAK - not relevant")
    assert llm.grade("q", [{"text": "x", "title": "t", "chunk_index": 0}]) == "weak"

def test_grade_parses_sufficient(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda messages, client=None: "SUFFICIENT")
    assert llm.grade("q", [{"text": "x", "title": "t", "chunk_index": 0}]) == "sufficient"

def test_decompose_parses_and_caps(monkeypatch):
    monkeypatch.setattr(llm, "chat",
                        lambda m, client=None: "sub one\nsub two\nsub three\nsub four")
    out = llm.decompose("compound q")
    assert out == ["sub one", "sub two", "sub three"]  # capped at 3

def test_decompose_failsafe_single(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda m, client=None: "   ")
    assert llm.decompose("atomic q") == ["atomic q"]

def test_synthesize_single_passthrough():
    sr = [{"sub_query": "q", "answer": "A", "citations": ["[T, chunk#0]"],
           "contexts": [], "path_trace": []}]
    out = llm.synthesize("q", sr)
    assert out == {"answer": "A", "citations": ["[T, chunk#0]"]}

def test_synthesize_merges_and_dedupes(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda m, client=None: "merged answer")
    sr = [
        {"sub_query": "q1", "answer": "a1", "citations": ["[T, chunk#0]"], "contexts": [], "path_trace": []},
        {"sub_query": "q2", "answer": "a2", "citations": ["[T, chunk#0]", "[web: x.com]"], "contexts": [], "path_trace": []},
    ]
    out = llm.synthesize("q", sr)
    assert out["answer"] == "merged answer"
    assert out["citations"] == ["[T, chunk#0]", "[web: x.com]"]

def test_synthesize_all_refusals():
    ref = {"sub_query": "q", "answer": "Not grounded in the corpus.",
           "citations": [], "contexts": [], "path_trace": []}
    out = llm.synthesize("q", [ref, dict(ref)])
    assert out["answer"] == "Not grounded in the corpus."
    assert out["citations"] == []
