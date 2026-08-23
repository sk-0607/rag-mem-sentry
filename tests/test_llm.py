import httpx
import pytest
from openai import RateLimitError

from app import config, llm


class _OkClient:
    """Minimal stand-in for the OpenAI client's success path."""
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                msg = type("M", (), {"content": "ok"})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def _rate_limited(headers=None):
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"),
                          headers=headers or {})
    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RateLimitError("slow down", response=resp, body=None)
    return Boom


def test_chat_backoff_grows_exponentially(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "x"}], client=_rate_limited())
    assert sleeps == sorted(sleeps)
    assert sleeps[-1] > sleeps[0]          # not a flat delay
    assert sum(sleeps) > 60                # outlasts a per-minute window

def test_chat_honours_retry_after_header(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "x"}],
                 client=_rate_limited({"retry-after": "7"}))
    assert all(s == pytest.approx(7.5) for s in sleeps)

def test_chat_paces_requests_when_interval_configured(monkeypatch):
    clock, sleeps = {"t": 100.0}, []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 4.0)
    monkeypatch.setattr(llm, "_last_call", 0.0)
    llm.chat([{"role": "user", "content": "x"}], client=_OkClient)   # first: no wait
    llm.chat([{"role": "user", "content": "x"}], client=_OkClient)   # immediate: waits
    assert sleeps == [pytest.approx(4.0)]

def test_chat_does_not_pace_by_default(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr(llm, "_last_call", 0.0)
    llm.chat([{"role": "user", "content": "x"}], client=_OkClient)
    llm.chat([{"role": "user", "content": "x"}], client=_OkClient)
    assert sleeps == []

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

def test_answer_prompt_tags_web_docs_by_domain(monkeypatch):
    seen = {}
    def fake_chat(messages, client=None):
        seen["ctx"] = messages[1]["content"]
        return "grounded answer"
    monkeypatch.setattr(llm, "chat", fake_chat)
    docs = [{"title": "PI 101", "chunk_index": 0, "text": "web body",
             "paper_id": "owasp.org", "source": "web"}]
    llm.answer("q", docs)
    assert "[web: owasp.org]" in seen["ctx"]
    assert "[PI 101, chunk#0]" not in seen["ctx"]

def test_answer_prompt_unchanged_when_experiences_empty(monkeypatch):
    captured = {}
    def fake_chat(messages, client=None):
        captured["content"] = messages[1]["content"]
        return "grounded answer"
    monkeypatch.setattr(llm, "chat", fake_chat)
    docs = [{"title": "T", "chunk_index": 0, "text": "ctx"}]
    llm.answer("q", docs)
    assert captured["content"] == f"Question: q\n\nContext:\n{llm._context_block(docs)}"

def test_answer_prompt_unchanged_when_experiences_explicitly_empty_string(monkeypatch):
    captured = {}
    def fake_chat(messages, client=None):
        captured["content"] = messages[1]["content"]
        return "grounded answer"
    monkeypatch.setattr(llm, "chat", fake_chat)
    docs = [{"title": "T", "chunk_index": 0, "text": "ctx"}]
    llm.answer("q", docs, experiences="")
    assert captured["content"] == f"Question: q\n\nContext:\n{llm._context_block(docs)}"

def test_answer_includes_experiences_block_when_present(monkeypatch):
    captured = {}
    def fake_chat(messages, client=None):
        captured["content"] = messages[1]["content"]
        return "grounded answer"
    monkeypatch.setattr(llm, "chat", fake_chat)
    docs = [{"title": "T", "chunk_index": 0, "text": "ctx"}]
    llm.answer("q", docs, experiences="Past experiences:\n- task: t\n  approach: a\n  outcome: o")
    assert "Past experiences:" in captured["content"]
    assert "approach: a" in captured["content"]

def test_answer_refusal_carries_no_citations(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda m, client=None: llm.REFUSAL)
    docs = [{"title": "T", "chunk_index": 0, "text": "x", "source": "corpus"}]
    out = llm.answer("q", docs)
    assert out["answer"] == llm.REFUSAL
    assert out["citations"] == []

def test_answer_refusal_with_trailing_tag_is_normalized(monkeypatch):
    # the model sometimes appends a stray citation tag after the refusal
    monkeypatch.setattr(llm, "chat",
                        lambda m, client=None: f"{llm.REFUSAL} [T, chunk#3]")
    docs = [{"title": "T", "chunk_index": 3, "text": "x", "source": "corpus"}]
    out = llm.answer("q", docs)
    assert out["answer"] == llm.REFUSAL
    assert out["citations"] == []

def test_answer_refusal_with_fullwidth_tag_is_normalized(monkeypatch):
    monkeypatch.setattr(llm, "chat",
                        lambda m, client=None: f"{llm.REFUSAL} 【T, chunk#3】")
    docs = [{"title": "T", "chunk_index": 3, "text": "x", "source": "corpus"}]
    assert llm.answer("q", docs) == {"answer": llm.REFUSAL, "citations": []}

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
