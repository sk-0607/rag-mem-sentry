"""Offline calibration: read saved screen rows, propose hard thresholds, report bands.

Writes nothing and never edits gate/policy.py; a person applies the proposal.
Only the calibration split is read, so a threshold cannot be tuned on the test
split by accident.
"""
import json
import sys
from pathlib import Path

from eval.fpr import dedupe_by_item_id
from eval.judge_fpr import top_signal
from gate.policy import DEFAULT_POLICY

PRIMARY_CLASSES = {"judge": (1, 3), "corroborate": (2,)}
BUDGET = 0.02

CALIBRATION = Path("data/phase4_screen/calibration/results.json")
CLEAN_EXPERIENCE = Path("data/phase4_screen/clean_experience/results.json")
S3_FPR = Path("data/phase4_s3_fpr/results.json")
R5_AUDIT = Path("data/phase3_r5/gate_audit.jsonl")


def signal_scores(rows: list[dict], signal: str) -> list[float]:
    return [s["score"] for r in rows for s in r["signals"] if s["signal"] == signal]


def propose_hard_threshold(attack_scores: list[float], clean_scores: list[float]) -> float | None:
    # Strictly above the clean maximum: a tie would block a clean item. Zero is
    # excluded so an abstaining-looking score is never proposed as a block line.
    if not attack_scores or not clean_scores:
        return None
    ceiling = max(clean_scores)
    above = [s for s in attack_scores if s > 0 and s > ceiling]
    return min(above) if above else None


def band_report(clean_rows: list[dict], attack_rows: list[dict], store: str,
                policy=DEFAULT_POLICY) -> dict:
    def usable(rows):
        return [r for r in rows if r["store"] == store and not r["reduced_assurance"]]

    clean, attack = usable(clean_rows), usable(attack_rows)
    current = policy.thresholds[store].quarantine_below
    non_admit = sum(r["action"] != "admit" for r in clean)
    rate = non_admit / len(clean) if clean else None
    within = None if rate is None else rate <= BUDGET

    largest = current
    if within is False:
        trusts = [r["trust"] for r in clean]
        passing = [q for q in set(trusts) | {current}
                   if q <= current and sum(t < q for t in trusts) / len(trusts) <= BUDGET]
        largest = max(passing)
    return {
        "store": store, "quarantine_below": current,
        "clean_total": len(clean), "clean_non_admit": non_admit, "clean_rate": rate,
        "budget": BUDGET, "within_budget": within,
        "attack_total": len(attack),
        "attack_caught": sum(r["action"] != "admit" for r in attack),
        "largest_band_within_budget": largest,
    }


def load_calibration_rows(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("split") != "calibration":
        raise ValueError(f"{path}: split is {data.get('split')!r}, not 'calibration'")
    return data["rows"]


def load_r5_rows(audit_path: Path) -> list[dict]:
    audit_path = Path(audit_path)
    if not audit_path.exists():
        return []
    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    read = dedupe_by_item_id([e for e in entries if e.get("stage") == "read"])
    return [{
        "id": e["item_id"], "class": "clean", "group": e["item_id"], "store": "corpus",
        "action": e["action"], "trust": e["trust"],
        "top_signal": top_signal(e) if e["signals"] else None,
        "signals": e["signals"],
        # The reason is the only trace of a degraded run (see read_verdict).
        "reduced_assurance": "reduced assurance" in e.get("reason", ""),
    } for e in read]


def _rows(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def main(calibration: Path = CALIBRATION, clean_experience: Path = CLEAN_EXPERIENCE,
         s3_fpr: Path = S3_FPR, r5: Path = R5_AUDIT) -> dict:
    calib = load_calibration_rows(calibration)
    r5_rows = load_r5_rows(r5)
    if not r5_rows:
        print("R5 audit missing or empty: R5 rows skipped", file=sys.stderr)
    s3_rows, exp_rows = _rows(s3_fpr), _rows(clean_experience)
    clean_by_signal = {"judge": r5_rows + s3_rows + exp_rows, "corroborate": s3_rows}

    out = {}
    for signal, classes in PRIMARY_CLASSES.items():
        attack = signal_scores([r for r in calib if r["class"] in classes], signal)
        clean = signal_scores(clean_by_signal[signal], signal)
        proposed = propose_hard_threshold(attack, clean)
        out[signal] = {"proposed": proposed, "clean_max": max(clean, default=None),
                       "clean_n": len(clean), "attack_n": len(attack),
                       "separable": proposed is not None}
    attack_rows = [r for r in calib if r["class"] != "clean"]
    for store in DEFAULT_POLICY.thresholds:
        out[store] = band_report(r5_rows + s3_rows + exp_rows, attack_rows, store)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
