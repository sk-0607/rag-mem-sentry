from fastapi.testclient import TestClient
import app.api as api

def test_health():
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}

def test_query(monkeypatch):
    monkeypatch.setattr(api.graph, "run",
                        lambda q: {"answer": "A", "citations": [], "path_trace": ["generate"]})
    client = TestClient(api.app)
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "A"
