import json
import threading
import time

from fastapi.testclient import TestClient

from app.web import create_app


def entry(item_id, stage="read", action="admit", signals=None):
    return {"ts": "t", "item_id": item_id, "store": "corpus", "stage": stage, "action": action,
            "trust": 1.0, "reason": "r", "signals": signals or []}


def result(answer="A", trace=()):
    return {"answer": answer, "citations": ["[doc-1]"], "path_trace": list(trace),
            "contexts": [], "experiences": [], "experiences_excluded": ["exp-9"]}


def test_index_serves_the_page():
    client = TestClient(create_app(lambda q: result(), None, lambda: {}))
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<title>" in r.text


def test_ask_returns_only_this_querys_read_rows(tmp_path):
    audit = tmp_path / "gate_audit.jsonl"
    audit.write_text(json.dumps(entry("earlier")) + "\n", encoding="utf-8")

    def runner(q):
        with audit.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry("now-1", action="block",
                                     signals=[{"signal": "judge", "score": 0.9, "reason": "why",
                                               "evidence_span": "bad"}])) + "\n")
            f.write(json.dumps(entry("now-w", stage="write")) + "\n")
        return result()

    client = TestClient(create_app(runner, audit, lambda: {"now-1": "a bad text"}))
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["answer"] == "A" and body["citations"] == ["[doc-1]"]
    assert body["gate_on"] is True and body["experiences_excluded"] == ["exp-9"]
    assert [r["item_id"] for r in body["rows"]] == ["now-1"]
    row = body["rows"][0]
    assert row["top_signal"] == "judge" and row["span_verbatim"] is True
    assert row["text"] == "a bad text"


def test_ask_with_gate_off_has_no_rows():
    client = TestClient(create_app(lambda q: result(), None, lambda: {}))
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["gate_on"] is False and body["rows"] == []


def test_reduced_assurance_comes_from_any_sub_query_trace():
    client = TestClient(create_app(lambda q: result(trace=["sq0:retrieve",
                                                           "sq1:gate:reduced_assurance"]),
                                   None, lambda: {}))
    assert client.post("/ask", json={"question": "q"}).json()["reduced_assurance"] is True
    client = TestClient(create_app(lambda q: result(trace=["sq0:retrieve"]), None, lambda: {}))
    assert client.post("/ask", json={"question": "q"}).json()["reduced_assurance"] is False


def test_blank_question_is_rejected_without_running():
    calls = []
    client = TestClient(create_app(lambda q: calls.append(q) or result(), None, lambda: {}))
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert calls == []


def test_queries_run_one_at_a_time():
    # The panel reads "audit entries since offset", which only holds for one query at a time.
    active, peak = [0], [0]

    def runner(q):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        return result()

    client = TestClient(create_app(runner, None, lambda: {}))
    threads = [threading.Thread(target=client.post, args=("/ask",),
                                kwargs={"json": {"question": "q"}}) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
