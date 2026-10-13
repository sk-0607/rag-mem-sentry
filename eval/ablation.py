"""Signal ablation derived offline from saved screen rows.

The corpus is screened once with every signal; each subset's action is recomputed
from the stored per-signal scores. Re-screening per subset would repeat the same
LLM calls, and a signal's score does not depend on which other signals run.
"""
import json
import sys
from pathlib import Path

from gate.policy import DEFAULT_POLICY
from gate.score import aggregate
from gate.types import SignalResult

SUBSETS = (
    ("structural",),
    ("judge",),
    ("corroborate",),
    ("structural", "judge"),
    ("structural", "judge", "corroborate"),
)
CLASSES = ("1", "2", "3")
_NOT_CHECKED = ("no_claims", "not_covered")


def recompute_action(signals: list[dict], subset: tuple[str, ...], store: str,
                     policy=DEFAULT_POLICY) -> str:
    # A subset signal missing from the row abstained; nothing is added for it.
    results = [SignalResult(s["signal"], s["score"], s["reason"])
               for s in signals if s["signal"] in subset]
    return policy.decide(aggregate(results), store).action.value


def ablation_table(rows: list[dict]) -> dict:
    table = {"+".join(s): {c: {"caught": 0, "scored": 0} for c in CLASSES} for s in SUBSETS}
    not_checked = {c: {"absent": 0, "no_claims": 0, "not_covered": 0} for c in CLASSES}
    excluded = 0
    for row in rows:
        if row["reduced_assurance"]:
            excluded += 1
            continue
        cls = str(row["class"])
        if cls not in CLASSES:
            continue
        corroborate = next((s for s in row["signals"] if s["signal"] == "corroborate"), None)
        if corroborate is None:
            not_checked[cls]["absent"] += 1
        elif corroborate["reason"] in _NOT_CHECKED:
            not_checked[cls][corroborate["reason"]] += 1
        for subset in SUBSETS:
            cell = table["+".join(subset)][cls]
            cell["scored"] += 1
            if recompute_action(row["signals"], subset, row["store"]) != "admit":
                cell["caught"] += 1
    return {"table": table, "excluded_reduced_assurance": excluded, "s3_not_checked": not_checked}


def markdown(out: dict) -> str:
    lines = ["| signals | class 1 | class 2 | class 3 |", "|---|---|---|---|"]
    for label, cells in out["table"].items():
        cols = " | ".join(f"{cells[c]['caught']}/{cells[c]['scored']}" for c in CLASSES)
        lines.append(f"| {label} | {cols} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m eval.ablation <results.json>", file=sys.stderr)
        return 2
    data = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = ablation_table(data["rows"])
    print(json.dumps(out, indent=2))
    print()
    print(markdown(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
