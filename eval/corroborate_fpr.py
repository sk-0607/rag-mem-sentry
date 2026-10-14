"""S3 false-positive rate on clean untrusted content.

Hold out some trusted sources, mark their chunks untrusted, and screen them
against the remaining trusted sources. Every non-admit S3 produces here is a
false positive by construction: the held-out text is clean.
"""
import json
import os
import random
import sys
from pathlib import Path

from app import config
from eval.attack_screen import row_from
from gate.read import default_read_gate
from gate.reference import ReferenceIndex
from gate.types import Item

OUT_DIR = Path("data/phase4_s3_fpr")
NOT_CHECKED = ("absent", "no_claims", "not_covered", "verbatim_trusted")
# Matches gate/signals/corroborate.py::_unparseable, which exposes no constant.
UNPARSEABLE = "corroborate reply could not be parsed"


def holdout_split(chunks: list[dict], n_sources: int, n_chunks: int,
                  seed: int = 0) -> tuple[list[dict], list[dict], list[str]]:
    trusted = [c for c in chunks if c.get("trusted") is True]
    sources = sorted({c["paper_id"] for c in trusted})
    if len(sources) < n_sources + 1:
        raise ValueError(f"need at least {n_sources + 1} trusted sources "
                         f"(at least one must remain as reference), found {len(sources)}")
    rng = random.Random(seed)
    picked = sorted(rng.sample(sources, n_sources))
    reference = [c for c in trusted if c["paper_id"] not in picked]
    pool = sorted((c for c in trusted if c["paper_id"] in picked), key=lambda c: c["id"])
    sample = pool if len(pool) <= n_chunks else rng.sample(pool, n_chunks)
    return reference, [{**c, "trusted": False} for c in sample], picked


def _reason(row: dict) -> str | None:
    return next((s["reason"] for s in row["signals"] if s["signal"] == "corroborate"), None)


def summarize(rows: list[dict]) -> dict:
    # A reduced-assurance row lost a signal, so its verdict says nothing clean
    # about S3; it is counted apart and kept out of every rate.
    kept = [r for r in rows if not r["reduced_assurance"]]
    out = {"total": len(kept), "reduced_assurance": len(rows) - len(kept),
           "non_admit": sum(r["action"] != "admit" for r in kept), "s3_checked": 0,
           "unparseable": 0,
           "s3_not_checked": dict.fromkeys(NOT_CHECKED, 0), "s3_topped_non_admit": 0,
           "s3_fpr": None, "reasons": {}}
    for r in kept:
        reason = _reason(r)
        if reason is not None:
            out["reasons"][reason] = out["reasons"].get(reason, 0) + 1
        if reason is not None and reason.startswith(UNPARSEABLE):
            # It quarantines, so it can be a false positive: it belongs in the denominator.
            out["unparseable"] += 1
            out["s3_checked"] += 1
        elif reason in ("supported", "contradicted"):
            out["s3_checked"] += 1
        else:
            out["s3_not_checked"][reason if reason in NOT_CHECKED else "absent"] += 1
        if r["action"] != "admit" and r["top_signal"] == "corroborate":
            out["s3_topped_non_admit"] += 1
    if out["s3_checked"]:
        out["s3_fpr"] = out["s3_topped_non_admit"] / out["s3_checked"]
    return out


def load_chunks(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str], chunks_path=None, out_dir=OUT_DIR, seed: int = 0) -> int:
    chunks = load_chunks(chunks_path if chunks_path is not None else config.CHUNKS_PATH)
    if not any(c.get("trusted") is True for c in chunks):
        print("no trusted chunks: re-ingest with trust flags first (Task 43)", file=sys.stderr)
        return 1
    n_sources = int(os.getenv("HOLDOUT_SOURCES", "4"))
    n_chunks = int(os.getenv("HOLDOUT_CHUNKS", "40"))
    holdout_file = os.getenv("HOLDOUT_FILE")
    if holdout_file:
        # Clean text from outside the index, screened against every trusted chunk:
        # on NQ, held-out sources share no claims, so S3 had nothing to check.
        pool = load_chunks(Path(holdout_file))
        held = pool if len(pool) <= n_chunks else random.Random(seed).sample(pool, n_chunks)
        reference = [c for c in chunks if c.get("trusted") is True]
        picked = sorted({c["paper_id"] for c in held})
    else:
        reference, held, picked = holdout_split(chunks, n_sources, n_chunks, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Not wiped: an interrupted run resumes from the cache.
    gate = default_read_gate(out_dir, reference=ReferenceIndex(reference))
    items = [Item(c["id"], c["text"], kind="corpus", metadata={"trusted": False}) for c in held]
    verdicts = gate.screen_batch(items, "corpus").verdicts
    rows = [{**row_from({"id": c["id"], "class": "clean", "group": c["paper_id"]}, "corpus", v),
             "paper_id": c["paper_id"]} for c, v in zip(held, verdicts)]
    path = out_dir / "results.json"
    path.write_text(json.dumps({"model": config.JUDGE_MODEL, "seed": seed,
                                "held_out_sources": picked, "holdout_file": holdout_file, "rows": rows,
                                "summary": summarize(rows)}, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
