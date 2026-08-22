"""Named seam for the experience store's gate.

Today this returns exactly `default_write_gate` unchanged — Phase 2 reuses
the Phase 0-1 gate as-is (S1 structural only, `corpus`/`experience`
thresholds from `DEFAULT_POLICY`, one shared audit/cache/quarantine under
`data_dir`). The seam exists so Phase 3 can add an experience-specific
rubric (e.g. an S2 signal tuned for agent-authored text) here without
touching the corpus ingest path.
"""
from pathlib import Path

from gate.screen import WriteGate, default_write_gate


def default_experience_gate(data_dir: Path) -> WriteGate:
    return default_write_gate(data_dir)
