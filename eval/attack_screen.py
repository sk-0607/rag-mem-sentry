"""Screen the attack corpus once through the read gate and save one row per entry.

The ablation (eval/ablation.py) is derived from these rows offline, so the LLM
signals run once per item, not once per signal subset.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from app import config
from eval.attack_corpus import corpus_dir, load, to_gate_item
from gate.read import ReadGate, default_read_gate
from gate.reference import default_reference

OUT_ROOT = Path("data/phase4_screen")


def row_from(entry: dict, store: str, verdict) -> dict:
    results = verdict.score.results
    return {
        "id": entry["id"], "class": entry["class"], "group": entry["group"], "store": store,
        "action": verdict.action.value, "trust": verdict.score.trust,
        "top_signal": max(results, key=lambda r: r.score).signal if results else None,
        "signals": [{"signal": r.signal, "score": r.score, "reason": r.reason} for r in results],
        "reduced_assurance": "reduced assurance" in verdict.reason,
    }


def screen(entries: list[dict], gate) -> list[dict]:
    by_store: dict[str, list] = {}
    for i, entry in enumerate(entries):
        item, store = to_gate_item(entry)
        by_store.setdefault(store, []).append((i, item))
    rows: list[dict | None] = [None] * len(entries)
    for store, pairs in by_store.items():
        result = gate.screen_batch([item for _, item in pairs], store)
        for (i, _), verdict in zip(pairs, result.verdicts):
            rows[i] = row_from(entries[i], store, verdict)
    return rows


def clean_experience_entries(path: Path) -> list[dict]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        entries.append({"id": rec["id"], "task": rec["task"], "approach": rec["approach"],
                        "outcome": rec["outcome"], "class": "clean", "group": rec["id"]})
    return entries


def build_gate(out_dir: Path, data_dir: Path) -> ReadGate:
    # Cache and audit under out_dir so a screen never mixes into the live gate's
    # files; the trusted set still comes from the real data directory.
    return default_read_gate(out_dir, reference=default_reference(data_dir))


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def policy_commit(git=_git) -> str:
    # Taken before screening and from the repo root, not the cwd. `-dirty` because
    # the last commit alone would label rows screened under an uncommitted edit.
    root = Path(__file__).resolve().parent.parent
    commit = git(["log", "-1", "--format=%H", "--", "gate/policy.py"], root).strip()
    if git(["status", "--porcelain", "--", "gate/policy.py"], root).strip():
        commit += "-dirty"
    return commit


def write_results(out_dir: Path, split: str, rows: list[dict], policy_commit: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps({"split": split, "model": config.JUDGE_MODEL,
                                "policy_commit": policy_commit, "rows": rows}, indent=2),
                    encoding="utf-8")
    return path


def main(argv: list[str], reference=None, gate_builder=build_gate, data_dir=None,
         out_root=OUT_ROOT) -> int:
    ap = argparse.ArgumentParser(prog="eval.attack_screen")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=("calibration", "test"))
    group.add_argument("--clean-experience", action="store_true")
    args = ap.parse_args(argv)

    data_dir = Path(data_dir) if data_dir is not None else Path(config.CHUNKS_PATH).parent
    if reference is None:
        reference = default_reference(data_dir)
    if len(reference) == 0:
        # With no trusted set S3 abstains on everything; the screen would report
        # an S3 column of zeros that means nothing.
        print("trusted reference is empty: re-ingest with trust flags first (Task 43)",
              file=sys.stderr)
        return 1

    if args.clean_experience:
        name, entries = "clean_experience", clean_experience_entries(Path(config.EXPERIENCES_PATH))
    else:
        name, entries = args.split, load(corpus_dir(), args.split)
    # Not wiped: after a daily-cap interruption the rerun must find the cache.
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = policy_commit()
    rows = screen(entries, gate_builder(out_dir, data_dir))
    print(write_results(out_dir, name, rows, commit))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
