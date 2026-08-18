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
    total = len(audit_entries)
    counts = {a: sum(1 for e in audit_entries if e["action"] == a) for a in ("admit", "quarantine", "block")}
    flagged = counts["quarantine"] + counts["block"]
    return {"total": total, **counts, "fpr": (flagged / total) if total else 0.0}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gate_audit.jsonl")
    stats = false_positive_rate(AuditLog(path).entries())
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
