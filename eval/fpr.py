"""False-positive measurement on the clean corpus.

The 22-paper AI-security corpus contains no real attacks — only quoted examples
inside papers that study them. So every non-admit verdict on this corpus is a
false positive by construction, which makes it a harder negative set than any
synthetic one.
"""
import json
import sys
from pathlib import Path

from gate.audit import AuditLog


def false_positive_rate(audit_entries: list[dict], stage: str = "write") -> dict:
    # Both stores append to one audit log, so this has to filter: an unfiltered
    # count silently folds experience verdicts into a number reported as the
    # corpus FPR. Symmetric with eval/experience_fpr.py's `experience` filter.
    # Stage defaults to "write" so this keeps measuring the write-time gate;
    # entries logged before the stage field existed have no "stage" key and
    # read as "write" too, so old audit logs are unaffected.
    entries = [
        e for e in audit_entries
        if e.get("store") == "corpus" and e.get("stage", "write") == stage
    ]
    total = len(entries)
    counts = {a: sum(1 for e in entries if e["action"] == a) for a in ("admit", "quarantine", "block")}
    flagged = counts["quarantine"] + counts["block"]
    return {"total": total, **counts, "fpr": (flagged / total) if total else 0.0}


def dedupe_by_item_id(audit_entries: list[dict]) -> list[dict]:
    """Keep only the last entry per item_id, in first-occurrence order.

    The audit log is append-only and never truncated, so re-running ingestion
    appends a second verdict per chunk under the same item_id. A later run's
    verdict supersedes an earlier one for the same content, so it's the one
    that should count toward the measurement.

    Deliberately absent from eval/experience_fpr.py: experience ids are uuid4,
    generated fresh per record, so no experience record is ever re-screened
    under an id it already has. Corpus chunk ids are content-derived and
    stable across re-ingests, which is what makes deduping necessary here.

    Keyed on store and stage too, not item_id alone: both stages share one
    log, and a later read-time verdict for a chunk is a different measurement,
    not a re-ingest, so it must not replace that chunk's write-time verdict.
    """
    last_by_id: dict[tuple, dict] = {}
    for entry in audit_entries:
        key = (entry.get("store"), entry.get("stage", "write"), entry["item_id"])
        last_by_id[key] = entry
    return list(last_by_id.values())


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gate_audit.jsonl")
    stats = false_positive_rate(dedupe_by_item_id(AuditLog(path).entries()))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
