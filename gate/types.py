from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Item:
    """A unit of content the gate screens: a corpus chunk or an experience record."""
    id: str
    text: str
    kind: str  # "corpus" | "experience"
    metadata: dict = field(default_factory=dict)


@dataclass
class GateContext:
    """What a signal is allowed to see besides the item itself."""
    store: str
    query: str | None = None
    siblings: list[Item] = field(default_factory=list)


@dataclass
class SignalResult:
    """One signal's opinion about one item. score is suspicion: 0.0 clean, 1.0 certain poison."""
    signal: str
    score: float
    reason: str
    evidence_span: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")


@dataclass
class Score:
    """Aggregated suspicion across signals. trust is 1.0 - aggregate suspicion."""
    trust: float
    top: SignalResult | None
    results: list[SignalResult]


class Action(Enum):
    ADMIT = "admit"
    QUARANTINE = "quarantine"
    BLOCK = "block"


@dataclass
class Verdict:
    action: Action
    score: Score
    reason: str
