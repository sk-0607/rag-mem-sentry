import pytest
from gate.types import Action, Item, Score, SignalResult, Verdict
from gate.quarantine import Quarantine


def held_verdict():
    result = SignalResult(signal="s1", score=0.5, reason="r")
    return Verdict(action=Action.QUARANTINE, score=Score(trust=0.5, top=result, results=[result]), reason="mid")


def test_held_item_is_retrievable_by_id(tmp_path):
    q = Quarantine(tmp_path / "q.jsonl")
    q.hold(Item(id="a", text="poison", kind="corpus"), held_verdict())
    assert [i.id for i in q.held()] == ["a"]


def test_held_item_preserves_text_and_metadata(tmp_path):
    q = Quarantine(tmp_path / "q.jsonl")
    q.hold(Item(id="a", text="poison", kind="corpus", metadata={"src": "paper"}), held_verdict())
    item = q.held()[0]
    assert item.text == "poison"
    assert item.metadata["src"] == "paper"


def test_release_returns_the_item_and_removes_it(tmp_path):
    q = Quarantine(tmp_path / "q.jsonl")
    q.hold(Item(id="a", text="poison", kind="corpus"), held_verdict())
    released = q.release("a")
    assert released.id == "a"
    assert q.held() == []


def test_release_of_unknown_id_raises(tmp_path):
    q = Quarantine(tmp_path / "q.jsonl")
    with pytest.raises(KeyError):
        q.release("nope")


def test_quarantine_survives_reopen(tmp_path):
    path = tmp_path / "q.jsonl"
    Quarantine(path).hold(Item(id="a", text="poison", kind="corpus"), held_verdict())
    assert len(Quarantine(path).held()) == 1
