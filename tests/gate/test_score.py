import pytest
from gate.types import SignalResult
from gate.score import aggregate


def r(name, score):
    return SignalResult(signal=name, score=score, reason=f"{name}@{score}")


def test_no_results_is_fully_trusted():
    score = aggregate([])
    assert score.trust == 1.0
    assert score.top is None
    assert score.results == []


def test_single_result_trust_is_complement_of_suspicion():
    assert aggregate([r("s1", 0.25)]).trust == pytest.approx(0.75)


def test_suspicion_compounds_across_signals():
    # noisy-OR: 1 - (1-0.5)(1-0.5) = 0.75 suspicion -> 0.25 trust
    assert aggregate([r("s1", 0.5), r("s2", 0.5)]).trust == pytest.approx(0.25)


def test_confident_signal_is_not_averaged_down_by_calm_ones():
    # a mean would give trust 0.4; noisy-OR keeps it low
    score = aggregate([r("s1", 0.9), r("s2", 0.0), r("s3", 0.0)])
    assert score.trust == pytest.approx(0.1)


def test_top_is_the_highest_suspicion_result():
    score = aggregate([r("s1", 0.2), r("s2", 0.7), r("s3", 0.5)])
    assert score.top.signal == "s2"


def test_certain_signal_drives_trust_to_zero():
    assert aggregate([r("s1", 1.0), r("s2", 0.1)]).trust == pytest.approx(0.0)


def test_results_are_preserved_in_order():
    results = [r("s1", 0.1), r("s2", 0.2)]
    assert aggregate(results).results == results
