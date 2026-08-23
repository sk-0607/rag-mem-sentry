"""Flag-rate measurement on generated experience records.

Unlike the corpus (eval/fpr.py), an experience store has no clean set that
exists independent of running the agent — it has to be generated. And unlike
the corpus, not every flag here is a false positive by construction: an
experience record *could* be genuinely worth quarantining. That's why this
reports `flag_rate`, not `fpr` — calling it a false-positive rate would
overclaim before every flagged record has been hand-read.
"""
import json
import sys
from pathlib import Path

from gate.audit import AuditLog


def flagged_experience_report(audit_entries: list[dict]) -> dict:
    entries = [e for e in audit_entries if e.get("store") == "experience"]
    total = len(entries)
    counts = {a: sum(1 for e in entries if e["action"] == a) for a in ("admit", "quarantine", "block")}
    flagged = counts["quarantine"] + counts["block"]
    return {"total": total, **counts, "flag_rate": (flagged / total) if total else 0.0}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gate_audit.jsonl")
    stats = flagged_experience_report(AuditLog(path).entries())
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
