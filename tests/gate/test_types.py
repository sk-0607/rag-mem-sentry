import pytest
from gate.types import Item, GateContext, SignalResult, Score, Verdict, Action


def test_signal_result_rejects_score_outside_unit_interval():
    with pytest.raises(ValueError):
        SignalResult(signal="s1", score=1.5, reason="x", evidence_span=None)
    with pytest.raises(ValueError):
        SignalResult(signal="s1", score=-0.1, reason="x", evidence_span=None)


def test_signal_result_accepts_boundaries():
    assert SignalResult(signal="s1", score=0.0, reason="x", evidence_span=None).score == 0.0
    assert SignalResult(signal="s1", score=1.0, reason="x", evidence_span=None).score == 1.0


def test_item_defaults_metadata_to_empty_dict():
    item = Item(id="a", text="hello", kind="corpus")
    assert item.metadata == {}


def test_item_metadata_is_not_shared_between_instances():
    a = Item(id="a", text="x", kind="corpus")
    b = Item(id="b", text="y", kind="corpus")
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_gate_context_defaults_are_write_time_shaped():
    ctx = GateContext(store="corpus")
    assert ctx.query is None
    assert ctx.siblings == []


def test_action_members():
    assert {a.name for a in Action} == {"ADMIT", "QUARANTINE", "BLOCK"}


def test_verdict_carries_action_and_score():
    score = Score(trust=0.9, top=None, results=[])
    verdict = Verdict(action=Action.ADMIT, score=score, reason="clean")
    assert verdict.action is Action.ADMIT
    assert verdict.score.trust == 0.9
