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
