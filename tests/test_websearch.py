import app.websearch as ws

class FakeDDGS:
    def __init__(self, results): self._results = results
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def text(self, query, max_results=3):
        return self._results[:max_results]

def test_web_search_shapes_docs(monkeypatch):
    fake = [{"title": "Prompt Injection 101", "href": "https://owasp.org/llm/pi",
             "body": "prompt injection is ..."}]
    monkeypatch.setattr(ws, "DDGS", lambda: FakeDDGS(fake))
    docs = ws.web_search("prompt injection", k=3)
    assert len(docs) == 1
    d = docs[0]
    assert d["source"] == "web"
    assert d["paper_id"] == "owasp.org"      # domain extracted for [web: domain]
    assert d["text"] == "prompt injection is ..."
    assert d["chunk_index"] == 0

def test_web_search_empty_on_error(monkeypatch):
    def boom(): raise RuntimeError("network")
    monkeypatch.setattr(ws, "DDGS", boom)
    assert ws.web_search("x") == []
