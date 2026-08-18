# tests/gate/test_policy.py
from gate.types import Action, Score, SignalResult
from gate.policy import GatePolicy, StoreThresholds


def score_of(*pairs):
    results = [SignalResult(signal=n, score=s, reason="t") for n, s in pairs]
    clean = 1.0
    for r in results:
        clean *= 1.0 - r.score
    top = max(results, key=lambda r: r.score) if results else None
    return Score(trust=clean, top=top, results=results)


POLICY = GatePolicy(
    thresholds={"corpus": StoreThresholds(block_below=0.1, quarantine_below=0.6)},
    hard_thresholds={"s1": 0.95},
)


def test_clean_item_is_admitted():
    assert POLICY.decide(score_of(("s1", 0.05)), "corpus").action is Action.ADMIT


def test_middling_trust_is_quarantined():
    assert POLICY.decide(score_of(("s1", 0.5)), "corpus").action is Action.QUARANTINE


def test_very_low_trust_is_blocked():
    assert POLICY.decide(score_of(("s1", 0.8), ("s2", 0.8)), "corpus").action is Action.BLOCK


def test_hard_threshold_blocks_regardless_of_aggregate():
    verdict = POLICY.decide(score_of(("s1", 0.96)), "corpus")
    assert verdict.action is Action.BLOCK
    assert "hard threshold" in verdict.reason


def test_hard_threshold_blocks_even_when_not_the_top_signal():
    # s1 crosses its hard threshold, but s2 (no hard threshold registered)
    # scores higher and would be score.top -- the override must still fire.
    verdict = POLICY.decide(score_of(("s1", 0.96), ("s2", 0.99)), "corpus")
    assert verdict.action is Action.BLOCK
    assert "hard threshold" in verdict.reason


def test_signal_without_a_hard_threshold_never_hard_blocks():
    # s2 has no entry in hard_thresholds; 0.99 suspicion alone lands in the band path
    verdict = POLICY.decide(score_of(("s2", 0.99)), "corpus")
    assert "hard threshold" not in verdict.reason


def test_unknown_store_quarantines_rather_than_admitting():
    verdict = POLICY.decide(score_of(("s1", 0.0)), "no-such-store")
    assert verdict.action is Action.QUARANTINE
    assert "unknown store" in verdict.reason


def test_verdict_carries_the_score_through():
    score = score_of(("s1", 0.5))
    assert POLICY.decide(score, "corpus").score is score
