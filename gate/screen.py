from dataclasses import dataclass
from pathlib import Path

from gate.audit import AuditLog
from gate.cache import VerdictCache, content_hash
from gate.policy import DEFAULT_POLICY, GatePolicy
from gate.quarantine import Quarantine
from gate.score import aggregate
from gate.signals.base import Signal, run_signals
from gate.signals.structural import StructuralSignal
from gate.types import Action, GateContext, Item, Verdict


@dataclass
class WriteGate:
    """The single write-time entry point. Nothing reaches a store without passing here."""

    signals: list[Signal]
    policy: GatePolicy
    cache: VerdictCache
    audit: AuditLog
    quarantine: Quarantine

    def __post_init__(self):
        # A gate with no signals aggregates to trust=1.0 and admits everything.
        # Default-deny must not depend on a caller remembering to pass signals.
        if not self.signals:
            raise ValueError("WriteGate requires at least one signal; refusing to admit-all")

    def screen(self, item: Item, store: str) -> Verdict:
        # Cache key includes store and the active signal set, not just text:
        # the same text under a different store or signal set is a different
        # question, and this keeps a future added signal (or a new store's
        # rubric) from silently reusing a verdict computed without it.
        signal_names = ",".join(sorted(s.name for s in self.signals))
        key = content_hash(f"{store}|{signal_names}|{item.text}")
        results = self.cache.get(key)
        failed: list[str] = []

        if results is None:
            run = run_signals(self.signals, item, GateContext(store=store))
            results, failed = run.results, run.failed
            # Never memoize a partial run — that would make a transient outage
            # permanent for this content.
            if not failed:
                self.cache.put(key, results)

        score = aggregate(results)
        if failed:
            # Spec: write-time failure fails closed. No user is waiting, nothing
            # is destroyed, and the item stays releasable.
            verdict = Verdict(
                action=Action.QUARANTINE,
                score=score,
                reason=f"signal(s) unavailable: {', '.join(failed)}; failing closed",
            )
        else:
            verdict = self.policy.decide(score, store)

        self.audit.record(item, verdict, store)
        if verdict.action is Action.QUARANTINE:
            self.quarantine.hold(item, verdict)
        return verdict


def default_write_gate(data_dir: Path) -> WriteGate:
    data_dir = Path(data_dir)
    return WriteGate(
        signals=[StructuralSignal()],
        policy=DEFAULT_POLICY,
        cache=VerdictCache(data_dir / "gate_cache.jsonl"),
        audit=AuditLog(data_dir / "gate_audit.jsonl"),
        quarantine=Quarantine(data_dir / "gate_quarantine.jsonl"),
    )
