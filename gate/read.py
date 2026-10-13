from dataclasses import dataclass
from pathlib import Path

from gate.audit import AuditLog
from gate.cache import VerdictCache, cache_key
from gate.policy import DEFAULT_POLICY, GatePolicy
from gate.score import aggregate
from gate.signals.base import Signal, run_signals_batch
from gate.signals.judge import JudgeSignal
from gate.signals.structural import StructuralSignal
from gate.types import GateContext, Item, Verdict


@dataclass
class ReadResult:
    """Verdicts for one screen_batch call, aligned with the input items."""

    verdicts: list[Verdict]
    reduced_assurance: bool


@dataclass
class ReadGate:
    """The read-time entry point. No quarantine field, and nothing here imports
    a module that writes a quarantine store — the absence is the enforcement.
    Where WriteGate fails closed on a signal failure (nothing is destroyed,
    no user is waiting), ReadGate degrades: a failed signal loses assurance,
    not the whole verdict, because a read-time caller IS waiting.
    """

    signals: list[Signal]
    policy: GatePolicy
    cache: VerdictCache
    audit: AuditLog
    batch_max: int = 5

    def __post_init__(self):
        # Same default-deny guard as WriteGate: an empty signal list must not
        # silently aggregate to trust=1.0 and admit everything.
        if not self.signals:
            raise ValueError("ReadGate requires at least one signal; refusing to admit-all")

    def screen_batch(self, items: list[Item], store: str) -> ReadResult:
        if not items:
            return ReadResult([], False)

        # Same cache-key scheme as WriteGate: store + active signal set + text.
        verdicts: list[Verdict | None] = [None] * len(items)
        misses: list[tuple[int, Item, str]] = []

        for idx, item in enumerate(items):
            key = cache_key(store, self.signals, item.text)
            results = self.cache.get(key)
            if results is None:
                misses.append((idx, item, key))
            else:
                verdicts[idx] = self.policy.decide(aggregate(results), store)

        reduced_assurance = False
        for start in range(0, len(misses), self.batch_max):
            chunk = misses[start:start + self.batch_max]
            chunk_items = [item for _, item, _ in chunk]
            # No query, no siblings: a read-time gate scores content on its
            # own merits, not against what else is in the batch.
            runs = run_signals_batch(self.signals, chunk_items, GateContext(store=store))

            for (idx, item, key), run in zip(chunk, runs):
                # Never memoize a partial run — same rule as WriteGate.
                if not run.failed:
                    self.cache.put(key, run.results)

                verdict = self.policy.decide(aggregate(run.results), store)
                if run.failed:
                    # Spec §7 "degrade, don't refuse": a read-time caller is
                    # waiting, so a failed signal reduces assurance rather
                    # than failing closed. When EVERY signal failed, results
                    # is empty, aggregate(...) gives trust 1.0, and
                    # policy.decide admits on that trust value with no
                    # special case needed here — the degraded-mode path
                    # already produces the right outcome.
                    reduced_assurance = True
                    verdict = Verdict(
                        action=verdict.action,
                        score=verdict.score,
                        reason=f"{verdict.reason}; reduced assurance: {', '.join(run.failed)} unavailable",
                    )
                verdicts[idx] = verdict

        for item, verdict in zip(items, verdicts):
            self.audit.record(item, verdict, store, stage="read")

        return ReadResult(verdicts=verdicts, reduced_assurance=reduced_assurance)


def default_read_gate(data_dir: Path, chat=None) -> ReadGate:
    # Lazy import, at call time not module load time: keeps gate/ importable
    # without app/ at import time, same discipline as JudgeSignal's own lazy
    # import of app.config/app.llm.
    from app import config

    data_dir = Path(data_dir)
    return ReadGate(
        signals=[StructuralSignal(), JudgeSignal(chat=chat)],
        policy=DEFAULT_POLICY,
        cache=VerdictCache(data_dir / "gate_cache.jsonl"),
        audit=AuditLog(data_dir / "gate_audit.jsonl"),
        batch_max=config.JUDGE_BATCH_MAX,
    )
