import logging
from dataclasses import dataclass, field
from typing import Protocol

from gate.types import GateContext, Item, SignalResult

log = logging.getLogger(__name__)


class Signal(Protocol):
    name: str

    def score(self, item: Item, ctx: GateContext) -> SignalResult | None:
        """Return suspicion for this item, or None if this signal does not apply."""
        ...


@dataclass
class SignalRun:
    """Outcome of running a signal set over one item.

    `failed` is carried separately because an empty `results` is ambiguous:
    it means either every signal looked and found nothing, or every signal
    crashed. Callers must be able to tell those apart.
    """
    results: list[SignalResult] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def run_signals(signals: list[Signal], item: Item, ctx: GateContext) -> SignalRun:
    """Run every signal, dropping ones that abstain and recording ones that fail.

    A signal fault is logged and excluded. It never propagates to the caller —
    a broken signal must not take down a query.
    """
    run = SignalRun()
    for signal in signals:
        try:
            result = signal.score(item, ctx)
        except Exception:
            log.warning("signal %s failed on item %s", signal.name, item.id, exc_info=True)
            run.failed.append(signal.name)
            continue
        if result is not None:
            run.results.append(result)
    return run


def run_signals_batch(signals: list[Signal], items: list[Item], ctx: GateContext) -> list[SignalRun]:
    """Same contract as run_signals, one SignalRun per item, in input order.

    A signal exposing score_batch (e.g. JudgeSignal, to fold N items into one
    LLM call instead of N) is called once for the whole list; a signal without
    one falls back to score() per item, wrapped exactly as run_signals wraps it.
    """
    runs = [SignalRun() for _ in items]
    for signal in signals:
        score_batch = getattr(signal, "score_batch", None)
        if score_batch is None:
            for run, item in zip(runs, items):
                try:
                    result = signal.score(item, ctx)
                except Exception:
                    log.warning("signal %s failed on item %s", signal.name, item.id, exc_info=True)
                    run.failed.append(signal.name)
                    continue
                if result is not None:
                    run.results.append(result)
            continue

        try:
            results = score_batch(items, ctx)
        except Exception:
            log.warning("signal %s failed on batch of %d items", signal.name, len(items), exc_info=True)
            for run in runs:
                run.failed.append(signal.name)
            continue
        if len(results) != len(items):
            log.warning(
                "signal %s returned %d results for %d items; treating as failed",
                signal.name, len(results), len(items),
            )
            for run in runs:
                run.failed.append(signal.name)
            continue
        for run, result in zip(runs, results):
            if result is not None:
                run.results.append(result)
    return runs
