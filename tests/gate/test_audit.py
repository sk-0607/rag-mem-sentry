from gate.types import Action, Item, Score, SignalResult, Verdict
from gate.audit import AuditLog


def verdict_of(action, signal_score=0.5):
    result = SignalResult(signal="s1", score=signal_score, reason="r", evidence_span="span")
    score = Score(trust=1.0 - signal_score, top=result, results=[result])
    return Verdict(action=action, score=score, reason="because")


def test_admits_are_recorded_too(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.ADMIT), "corpus")
    assert log.entries()[0]["action"] == "admit"


def test_entry_carries_signal_evidence(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.BLOCK), "corpus")
    entry = log.entries()[0]
    assert entry["item_id"] == "a"
    assert entry["store"] == "corpus"
    assert entry["trust"] == 0.5
    assert entry["signals"][0]["evidence_span"] == "span"


def test_entries_are_append_only_and_ordered(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.ADMIT), "corpus")
    log.record(Item(id="b", text="y", kind="corpus"), verdict_of(Action.BLOCK), "corpus")
    assert [e["item_id"] for e in log.entries()] == ["a", "b"]


def test_entries_survive_reopen(tmp_path):
    path = tmp_path / "a.jsonl"
    AuditLog(path).record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.ADMIT), "corpus")
    assert len(AuditLog(path).entries()) == 1


def test_record_defaults_stage_to_write(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.ADMIT), "corpus")
    assert log.entries()[0]["stage"] == "write"


def test_record_stores_read_stage(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.record(Item(id="a", text="x", kind="corpus"), verdict_of(Action.ADMIT), "corpus", stage="read")
    assert log.entries()[0]["stage"] == "read"
