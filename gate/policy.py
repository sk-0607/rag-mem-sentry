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
        if score.top is not None:
            hard = self.hard_thresholds.get(score.top.signal)
            if hard is not None and score.top.score >= hard:
                return Verdict(
                    action=Action.BLOCK,
                    score=score,
                    reason=f"{score.top.signal} over hard threshold ({score.top.score:.2f} >= {hard:.2f})",
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


# Placeholder values. Task 11 replaces these with figures calibrated against
# the measured false-positive rate on the clean hard-negative set.
DEFAULT_POLICY = GatePolicy(
    thresholds={
        "corpus": StoreThresholds(block_below=0.10, quarantine_below=0.60),
        "experience": StoreThresholds(block_below=0.20, quarantine_below=0.70),
    },
    hard_thresholds={"structural": 0.93},
)
