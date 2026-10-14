import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from eval import probe_server
from eval.attack_corpus import to_gate_item
from eval.probe_server import (CapReached, CountingChat, ProbeError, ProbeLog, create_app,
                               parse_probe)
from gate.audit import AuditLog
from gate.cache import VerdictCache
from gate.policy import DEFAULT_POLICY
from gate.read import ReadGate, ReadResult
from gate.score import aggregate
from gate.types import SignalResult, Verdict

CORPUS = {"class": 1, "store": "corpus", "text": "The river crosses three counties."}
EXPERIENCE = {"class": 3, "store": "experience", "task": "sort a list",
              "approach": "used the built-in sort", "outcome": "list sorted"}


class FakeChat:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages, client=None, model=None):
        self.calls += 1
        return "{}"


class FakeGate:
    """Cache keyed by text; one chat call per miss; suspicion set per text."""

    def __init__(self, counter):
        self.counter = counter
        self.cache = {}
        self.scores = {}
        self.reduced = False
        self.seen = []

    def screen_batch(self, items, store):
        self.seen.append((items, store))
        item = items[0]
        if item.text not in self.cache:
            self.counter([{"role": "user", "content": item.text}])
            results = [SignalResult("judge", self.scores.get(item.text, 0.0), "fake")]
            if self.reduced:
                verdict = DEFAULT_POLICY.decide(aggregate(results), store)
                v = Verdict(verdict.action, verdict.score, f"{verdict.reason}; reduced assurance: judge unavailable")
                return ReadResult([v], True)
            self.cache[item.text] = results
        return ReadResult([DEFAULT_POLICY.decide(aggregate(self.cache[item.text]), store)], False)


def make(tmp_path, cap=20, session=1, gate=None):
    chat = FakeChat()
    counter = CountingChat(chat)
    gate = gate or FakeGate(counter)
    log = ProbeLog(tmp_path / "probe" / "probes.jsonl")
    client = TestClient(create_app(gate, counter, log, session, cap))
    return client, gate, chat, log


def rows(log):
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


def corpus(text):
    return {"class": 2, "store": "corpus", "text": text}


# --- parse -----------------------------------------------------------------

def test_parse_accepts_each_class():
    for body in (CORPUS, corpus("x"), EXPERIENCE):
        entry = parse_probe(dict(body))
        assert entry["id"].startswith("probe-") and entry["class"] == body["class"]


def test_parse_rejects_non_object_body():
    for body in ([CORPUS], "text", None, 3):
        with pytest.raises(ProbeError):
            parse_probe(body)


def test_parse_rejects_unknown_class():
    for cls in (4, "2"):
        with pytest.raises(ProbeError):
            parse_probe({**CORPUS, "class": cls})


def test_parse_rejects_class_store_mismatch():
    with pytest.raises(ProbeError):
        parse_probe({**CORPUS, "store": "experience"})
    with pytest.raises(ProbeError):
        parse_probe({**EXPERIENCE, "store": "corpus"})


def test_parse_rejects_trusted_key():
    for value in (True, False):
        with pytest.raises(ProbeError, match="trusted"):
            parse_probe({**CORPUS, "trusted": value})


def test_parse_rejects_unknown_key():
    with pytest.raises(ProbeError):
        parse_probe({**CORPUS, "note": "x"})


def test_parse_rejects_missing_or_empty_field():
    with pytest.raises(ProbeError):
        parse_probe({"class": 1, "store": "corpus"})
    with pytest.raises(ProbeError):
        parse_probe({**CORPUS, "text": ""})
    with pytest.raises(ProbeError):
        parse_probe({**EXPERIENCE, "outcome": 5})


def test_parse_rejects_oversize_text():
    with pytest.raises(ProbeError):
        parse_probe({**CORPUS, "text": "a" * 8001})
    assert parse_probe({**CORPUS, "text": "a" * 8000})


def test_probe_id_is_stable_for_same_fields():
    assert parse_probe(dict(CORPUS))["id"] == parse_probe(dict(CORPUS))["id"]
    assert parse_probe(dict(CORPUS))["id"] != parse_probe(corpus("other"))["id"]


# --- requests --------------------------------------------------------------

def test_bad_request_is_400_and_not_logged(tmp_path):
    client, gate, _, log = make(tmp_path)
    assert client.post("/probe", json={**CORPUS, "trusted": True}).status_code == 400
    assert client.post("/probe", content=b"not json",
                       headers={"content-type": "application/json"}).status_code == 400
    assert gate.seen == [] and log.counted(1) == 0 and not log.path.exists()


def test_corpus_probe_reaches_gate_untrusted_as_corpus(tmp_path):
    client, gate, _, _ = make(tmp_path)
    client.post("/probe", json=CORPUS)
    (items, store), = gate.seen
    assert store == "corpus" and items[0].metadata["trusted"] is False


def test_experience_probe_projects_like_attack_corpus(tmp_path):
    client, gate, _, _ = make(tmp_path)
    client.post("/probe", json=EXPERIENCE)
    (items, store), = gate.seen
    expected, _ = to_gate_item({**EXPERIENCE, "id": "x"})
    assert store == "experience" and items[0].text == expected.text


def test_response_has_only_admitted_probes_left_session(tmp_path):
    client, _, _, _ = make(tmp_path)
    r = client.post("/probe", json=CORPUS)
    assert r.status_code == 200
    assert r.json() == {"admitted": True, "probes_left": 19, "session": 1}


def test_quarantine_and_block_both_return_admitted_false(tmp_path):
    client, gate, _, _ = make(tmp_path)
    gate.scores = {"q": 0.5, "b": 0.95}
    assert client.post("/probe", json=corpus("q")).json()["admitted"] is False
    assert client.post("/probe", json=corpus("b")).json()["admitted"] is False


def test_uncached_probe_counted(tmp_path):
    client, _, chat, log = make(tmp_path)
    client.post("/probe", json=CORPUS)
    assert chat.calls == 1 and log.counted(1) == 1


def test_cached_resubmission_not_counted(tmp_path):
    client, _, chat, log = make(tmp_path)
    client.post("/probe", json=CORPUS)
    r = client.post("/probe", json=CORPUS)
    assert r.json()["probes_left"] == 19 and chat.calls == 1 and log.counted(1) == 1
    assert rows(log)[-1]["cached"] is True and rows(log)[-1]["counted"] is False


def test_cap_returns_429_for_uncached_and_spends_no_calls(tmp_path):
    client, _, chat, log = make(tmp_path, cap=1)
    client.post("/probe", json=corpus("a"))
    r = client.post("/probe", json=corpus("b"))
    assert r.status_code == 429 and chat.calls == 1 and log.counted(1) == 1


def test_cap_still_answers_cached_items(tmp_path):
    client, _, _, _ = make(tmp_path, cap=1)
    client.post("/probe", json=corpus("a"))
    r = client.post("/probe", json=corpus("a"))
    assert r.status_code == 200 and r.json()["probes_left"] == 0


class ChatSignal:
    name = "judge"

    def __init__(self, chat):
        self.chat = chat

    def score(self, item, ctx):
        self.chat([{"role": "user", "content": item.text}])
        return SignalResult(self.name, 0.0, "fake")


def real_gate(directory, counter):
    return ReadGate(signals=[ChatSignal(counter)], policy=DEFAULT_POLICY,
                    cache=VerdictCache(directory / "gate_cache.jsonl"),
                    audit=AuditLog(directory / "gate_audit.jsonl"))


def test_refusing_chat_leaves_no_cache_entry(tmp_path):
    chat = FakeChat()
    counter = CountingChat(chat)
    probe_dir = tmp_path / "probe"
    gate = real_gate(probe_dir, counter)
    log = ProbeLog(probe_dir / "probes.jsonl")
    client = TestClient(create_app(gate, counter, log, 1, 1))
    client.post("/probe", json=corpus("a"))
    assert client.post("/probe", json=corpus("b")).status_code == 429
    cache = (probe_dir / "gate_cache.jsonl").read_text(encoding="utf-8")
    assert len(cache.splitlines()) == 1 and chat.calls == 1


def test_reduced_assurance_returns_503_uncounted_and_logs_calls(tmp_path):
    client, gate, _, log = make(tmp_path)
    gate.reduced = True
    assert client.post("/probe", json=CORPUS).status_code == 503
    row = rows(log)[-1]
    assert row["counted"] is False and row["admitted"] is None and row["calls"] == 1
    assert log.counted(1) == 0


def test_restart_resumes_session_count(tmp_path):
    client, _, _, log = make(tmp_path)
    client.post("/probe", json=CORPUS)
    again = TestClient(create_app(FakeGate(CountingChat(FakeChat())), CountingChat(FakeChat()),
                                  ProbeLog(log.path), 1, 20))
    assert again.get("/status").json()["probes_left"] == 19


def test_session_counts_are_separate(tmp_path):
    client, _, _, log = make(tmp_path)
    client.post("/probe", json=CORPUS)
    other = TestClient(create_app(FakeGate(CountingChat(FakeChat())), CountingChat(FakeChat()),
                                  ProbeLog(log.path), 2, 20))
    assert other.get("/status").json() == {"session": 2, "probes_left": 20}


def test_status_route(tmp_path):
    client, _, _, _ = make(tmp_path)
    assert client.get("/status").json() == {"session": 1, "probes_left": 20}


def test_only_probe_and_status_routes_exist(tmp_path):
    client, _, _, _ = make(tmp_path)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_log_row_fields(tmp_path):
    client, gate, _, log = make(tmp_path, cap=1)
    client.post("/probe", json=corpus("a"))
    client.post("/probe", json=corpus("b"))
    gate.reduced = True
    gate.counter.refuse = False
    client2 = TestClient(create_app(gate, gate.counter, log, 2, 20))
    client2.post("/probe", json=corpus("c"))
    keys = {"ts", "session", "class", "store", "item_hash", "fields", "status", "cached",
            "counted", "calls", "admitted", "action"}
    got = rows(log)
    assert [r["status"] for r in got] == [200, 429, 503]
    assert all(set(r) == keys for r in got)
    assert got[0]["item_hash"] == hashlib.sha256(b"a").hexdigest()
    assert got[0]["fields"] == {"text": "a"} and got[0]["action"] == "ADMIT"
    assert got[1]["action"] is None and got[2]["action"] is None


# --- main ------------------------------------------------------------------

class FakeReference:
    def __init__(self, n=1):
        self.n = n

    def __len__(self):
        return self.n


@pytest.fixture
def wired(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setenv("PROBE_SESSION", "1")
    monkeypatch.setattr(probe_server, "PROBE_DIR", str(tmp_path / "probe"))
    monkeypatch.setattr(probe_server, "require_frozen", lambda: None)
    monkeypatch.setattr(probe_server, "default_reference", lambda data_dir: FakeReference())

    def fake_read_gate(data_dir, chat=None, reference=None):
        calls["gate"] = (data_dir, chat, reference)
        return FakeGate(chat)

    monkeypatch.setattr(probe_server, "default_read_gate", fake_read_gate)
    monkeypatch.setattr(probe_server.uvicorn, "run", lambda app, **kw: calls.setdefault("run", kw))
    return calls


def test_main_refuses_bad_session(wired, monkeypatch):
    monkeypatch.delenv("PROBE_SESSION")
    with pytest.raises(SystemExit) as e:
        probe_server.main()
    assert e.value.code
    for bad in ("0", "4"):
        monkeypatch.setenv("PROBE_SESSION", bad)
        with pytest.raises(SystemExit):
            probe_server.main()
    assert "gate" not in wired


def test_main_refuses_freeze_difference_before_building_gate(wired, monkeypatch):
    def boom():
        raise probe_server.FreezeError("gate differs")

    monkeypatch.setattr(probe_server, "require_frozen", boom)
    with pytest.raises(SystemExit) as e:
        probe_server.main()
    assert "gate differs" in str(e.value.code) and "gate" not in wired


def test_main_refuses_empty_trusted_set(wired, monkeypatch):
    monkeypatch.setattr(probe_server, "default_reference", lambda data_dir: FakeReference(0))
    with pytest.raises(SystemExit) as e:
        probe_server.main()
    assert "Task 43" in str(e.value.code) and "gate" not in wired


def test_main_wires_counter_real_reference_and_probe_dir(wired, tmp_path):
    probe_server.main()
    data_dir, chat, reference = wired["gate"]
    assert isinstance(chat, CountingChat) and reference is not None
    assert str(data_dir) == str(tmp_path / "probe")


def test_main_binds_localhost(wired):
    probe_server.main()
    assert wired["run"]["host"] == "127.0.0.1"


def test_writes_only_under_probe_dir(tmp_path):
    chat = FakeChat()
    counter = CountingChat(chat)
    probe_dir = tmp_path / "probe"
    client = TestClient(create_app(real_gate(probe_dir, counter), counter,
                                   ProbeLog(probe_dir / "probes.jsonl"), 1, 20))
    for body in (CORPUS, corpus("y"), EXPERIENCE):
        assert client.post("/probe", json=body).status_code == 200
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and all(probe_dir in p.parents for p in files)


def test_counting_chat_refuses_without_calling():
    chat = FakeChat()
    counter = CountingChat(chat)
    counter.refuse = True
    with pytest.raises(CapReached):
        counter([{"role": "user", "content": "x"}])
    assert chat.calls == 0 and counter.refused and counter.calls == 0
    counter.reset()
    assert not counter.refused and counter.calls == 0
