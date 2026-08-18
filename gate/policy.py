from dataclasses import dataclass

from gate.types import Action, Score, Verdict


@dataclass(frozen=True)
class StoreThresholds:
    """Trust cut points for one store. trust below block_below is blocked,
    below quarantine_below is quarantined, otherwise admitted."""
    block_below: float
    quarantine_below: float


@dataclass(frozen=True)
class GatePolicy:
    thresholds: dict[str, StoreThresholds]
    hard_thresholds: dict[str, float]

    def decide(self, score: Score, store: str) -> Verdict:
        for result in score.results:
            hard = self.hard_thresholds.get(result.signal)
            if hard is not None and result.score >= hard:
                return Verdict(
                    action=Action.BLOCK,
                    score=score,
                    reason=f"{result.signal} over hard threshold ({result.score:.2f} >= {hard:.2f})",
                )

        limits = self.thresholds.get(store)
        if limits is None:
            return Verdict(
                action=Action.QUARANTINE,
                score=score,
                reason=f"unknown store {store!r}; quarantining rather than admitting",
            )

        if score.trust < limits.block_below:
            return Verdict(action=Action.BLOCK, score=score, reason=f"trust {score.trust:.2f} below block floor")
        if score.trust < limits.quarantine_below:
            return Verdict(action=Action.QUARANTINE, score=score, reason=f"trust {score.trust:.2f} below quarantine floor")
        return Verdict(action=Action.ADMIT, score=score, reason=f"trust {score.trust:.2f} clears thresholds")


# `corpus` thresholds are measured, not placeholder: Task 11 ran the write
# gate over the real 22-paper AI-security corpus (2,025 chunks) and got
# admit=2017, quarantine=8, block=0 -> FPR 0.395%, against a 2% budget. All 8
# quarantined chunks were hand-read; every one is a genuine quoted attack/
# red-team example (indirect-injection payload, GCG jailbreak illustration,
# harmful-fine-tuning system prompt, or Anthropic sleeper-agents red-team
# prompt) rather than a regex over-match on ordinary prose, so block_below and
# quarantine_below are kept at their original values and treated as
# calibrated for this corpus. See README.md "False-positive measurement" for
# the full read-through. `experience` remains an unmeasured placeholder —
# Phase 2 (Task 16) exercises the experience store for the first time.
DEFAULT_POLICY = GatePolicy(
    thresholds={
        "corpus": StoreThresholds(block_below=0.10, quarantine_below=0.60),
        "experience": StoreThresholds(block_below=0.20, quarantine_below=0.70),
    },
    # structural's 0.93 is derived from StructuralSignal's pattern weights
    # (gate/signals/structural.py): above two instruction patterns (0.70), at
    # or below two patterns plus an obfuscation hit (0.94). Recompute if
    # either weight changes. This dict, not any attribute on the Signal
    # instance, is what GatePolicy.decide() actually reads.
    hard_thresholds={"structural": 0.93},
)
