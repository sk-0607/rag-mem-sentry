import logging
from gate.types import Item, GateContext, SignalResult
from gate.signals.base import SignalRun, run_signals, run_signals_batch


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


ITEMS = [
    Item(id="a", text="hello", kind="corpus"),
    Item(id="b", text="world", kind="corpus"),
]


class BatchStubSignal:
    """A signal with score_batch, e.g. JudgeSignal folding N items into one LLM call."""
    name = "batch"

    def __init__(self, results):
        self._results = results
        self.calls = 0

    def score_batch(self, items, ctx):
        self.calls += 1
        return self._results


class RaisingBatchSignal:
    name = "batch-boom"

    def score_batch(self, items, ctx):
        raise RuntimeError("batch exploded")


class WrongLengthBatchSignal:
    name = "batch-short"

    def score_batch(self, items, ctx):
        return [SignalResult(signal="batch-short", score=0.1, reason="only one")]


def test_batch_signal_called_once_for_all_items():
    a = SignalResult(signal="batch", score=0.1, reason="a")
    b = SignalResult(signal="batch", score=0.2, reason="b")
    signal = BatchStubSignal([a, b])
    runs = run_signals_batch([signal], ITEMS, CTX)
    assert signal.calls == 1
    assert [r.results for r in runs] == [[a], [b]]


def test_batch_signal_raise_marks_every_item_failed():
    runs = run_signals_batch([RaisingBatchSignal()], ITEMS, CTX)
    assert [r.failed for r in runs] == [["batch-boom"], ["batch-boom"]]
    assert [r.results for r in runs] == [[], []]


def test_batch_signal_wrong_length_marks_every_item_failed():
    runs = run_signals_batch([WrongLengthBatchSignal()], ITEMS, CTX)
    assert [r.failed for r in runs] == [["batch-short"], ["batch-short"]]
    assert [r.results for r in runs] == [[], []]


def test_batch_signal_none_entry_abstains_not_failed():
    b = SignalResult(signal="batch", score=0.2, reason="b")
    signal = BatchStubSignal([None, b])
    runs = run_signals_batch([signal], ITEMS, CTX)
    assert runs[0].results == [] and runs[0].failed == []
    assert runs[1].results == [b]


def test_per_item_signal_used_when_no_score_batch():
    batch_a = SignalResult(signal="batch", score=0.1, reason="a")
    batch_b = SignalResult(signal="batch", score=0.2, reason="b")
    batch_signal = BatchStubSignal([batch_a, batch_b])
    per_item = {
        "a": SignalResult(signal="stub", score=0.3, reason="pa"),
        "b": SignalResult(signal="stub", score=0.4, reason="pb"),
    }

    class PerItemSignal:
        name = "stub"

        def score(self, item, ctx):
            return per_item[item.id]

    runs = run_signals_batch([batch_signal, PerItemSignal()], ITEMS, CTX)
    assert runs[0].results == [batch_a, per_item["a"]]
    assert runs[1].results == [batch_b, per_item["b"]]


def test_per_item_raise_fails_only_that_item():
    class RaisesOnA:
        name = "boom"

        def score(self, item, ctx):
            if item.id == "a":
                raise RuntimeError("boom")
            return None

    runs = run_signals_batch([RaisesOnA()], ITEMS, CTX)
    assert runs[0].failed == ["boom"]
    assert runs[1].failed == []


def test_runs_align_with_input_order():
    runs = run_signals_batch([], ITEMS, CTX)
    assert len(runs) == len(ITEMS)
    assert all(isinstance(r, SignalRun) for r in runs)
