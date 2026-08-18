import logging
from gate.types import Item, GateContext, SignalResult
from gate.signals.base import run_signals


class StubSignal:
    name = "stub"
    hard_threshold = 0.9

    def __init__(self, result):
        self._result = result

    def score(self, item, ctx):
        return self._result


class RaisingSignal:
    name = "boom"
    hard_threshold = 0.9

    def score(self, item, ctx):
        raise RuntimeError("signal exploded")


ITEM = Item(id="a", text="hello", kind="corpus")
CTX = GateContext(store="corpus")


def test_collects_results_from_all_signals():
    a = SignalResult(signal="stub", score=0.2, reason="a")
    b = SignalResult(signal="stub", score=0.4, reason="b")
    assert run_signals([StubSignal(a), StubSignal(b)], ITEM, CTX).results == [a, b]


def test_drops_none_results():
    a = SignalResult(signal="stub", score=0.2, reason="a")
    assert run_signals([StubSignal(a), StubSignal(None)], ITEM, CTX).results == [a]


def test_abstaining_signal_is_not_recorded_as_failed():
    assert run_signals([StubSignal(None)], ITEM, CTX).failed == []


def test_raising_signal_is_excluded_not_propagated():
    a = SignalResult(signal="stub", score=0.2, reason="a")
    assert run_signals([RaisingSignal(), StubSignal(a)], ITEM, CTX).results == [a]


def test_raising_signal_is_named_in_failed():
    assert run_signals([RaisingSignal()], ITEM, CTX).failed == ["boom"]


def test_raising_signal_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        run_signals([RaisingSignal()], ITEM, CTX)
    assert "boom" in caplog.text


def test_all_signals_failing_is_distinguishable_from_all_clean():
    all_failed = run_signals([RaisingSignal(), RaisingSignal()], ITEM, CTX)
    all_clean = run_signals([StubSignal(None), StubSignal(None)], ITEM, CTX)
    assert all_failed.results == all_clean.results == []
    assert all_failed.failed == ["boom", "boom"]
    assert all_clean.failed == []
