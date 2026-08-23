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


def false_positive_rate(audit_entries: list[dict]) -> dict:
    # Both stores append to one audit log, so this has to filter: an unfiltered
    # count silently folds experience verdicts into a number reported as the
    # corpus FPR. Symmetric with eval/experience_fpr.py's `experience` filter.
    entries = [e for e in audit_entries if e.get("store") == "corpus"]
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
    """
    last_by_id: dict[str, dict] = {}
    for entry in audit_entries:
        last_by_id[entry["item_id"]] = entry
    return list(last_by_id.values())


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gate_audit.jsonl")
    stats = false_positive_rate(dedupe_by_item_id(AuditLog(path).entries()))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
