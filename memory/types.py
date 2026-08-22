from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from gate.types import Item


@dataclass
class ExperienceRecord:
    """An agent's experience record: task attempted, approach taken, outcome achieved."""
    id: str
    task: str
    approach: str
    outcome: str
    success: bool
    ts: str

    def __post_init__(self):
        if not self.task:
            raise ValueError("task must not be empty")

    def screened_text(self) -> str:
        """Return concatenated free-text fields for screening.
        
        Labels ensure fields don't run together into unintended meanings.
        """
        return f"task: {self.task}\napproach: {self.approach}\noutcome: {self.outcome}"

    def to_item(self) -> Item:
        """Project into gate's Item for screening."""
        return Item(
            id=self.id,
            text=self.screened_text(),
            kind="experience",
            metadata={
                "success": self.success,
                "ts": self.ts,
            }
        )


def record_from_run(task: str, approach: str, outcome: str, success: bool) -> ExperienceRecord:
    """Factory: create an ExperienceRecord with minted id and timestamp."""
    return ExperienceRecord(
        id=uuid4().hex,
        task=task,
        approach=approach,
        outcome=outcome,
        success=success,
        ts=datetime.now(timezone.utc).isoformat(),
    )
