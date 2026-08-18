import logging
from dataclasses import dataclass, field
from typing import Protocol

from gate.types import GateContext, Item, SignalResult

log = logging.getLogger(__name__)


class Signal(Protocol):
    name: str
    hard_threshold: float

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
