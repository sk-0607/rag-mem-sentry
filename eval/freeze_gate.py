"""Record the gate's verdict-bearing state, and check the current state against it.

    python -m eval.freeze_gate record   # writes eval/frozen_gate.json; no LLM call
    python -m eval.freeze_gate check    # exit 0 when it matches, else lists every difference, exit 1

A content record, not a commit check: comparing HEAD to a tag fails on every
unrelated docs commit and passes a dirty tree or a changed `.env` (Phase 5 spec §4).
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from app import config
from gate.policy import DEFAULT_POLICY
from gate.reference import default_reference

FROZEN_GATE = Path("eval/frozen_gate.json")
KNOBS = ("JUDGE_BATCH_MAX", "S3_MAX_CLAIMS", "S3_REF_K")


class FreezeError(RuntimeError):
    pass


def real_data_dir() -> Path:
    # The directory eval/attack_screen.py::main reads the trusted set from.
    return Path(config.CHUNKS_PATH).parent


def _git_lines(root: Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def gate_files(root: Path) -> dict[str, str]:
    # Tracked plus new-but-unignored: a fourth signal dropped in and not yet
    # committed must still count as a change; __pycache__ must not.
    paths = set(_git_lines(root, "ls-files", "gate"))
    paths |= set(_git_lines(root, "ls-files", "--others", "--exclude-standard", "gate"))
    return {p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest()
            for p in sorted(paths) if (Path(root) / p).is_file()}


def policy_values(policy=None) -> dict:
    policy = policy if policy is not None else DEFAULT_POLICY
    return {
        "thresholds": {store: {"block_below": t.block_below, "quarantine_below": t.quarantine_below}
                       for store, t in sorted(policy.thresholds.items())},
        "hard_thresholds": dict(sorted(policy.hard_thresholds.items())),
        "quarantine_only": sorted(policy.quarantine_only),
    }


def gate_state(data_dir: Path, root: Path = Path(".")) -> dict:
    return {
        "gate_files": gate_files(Path(root)),
        "knobs": {k: getattr(config, k) for k in KNOBS},
        "judge_model": config.JUDGE_MODEL,
        "policy": policy_values(DEFAULT_POLICY),
        "reference_fingerprint": default_reference(Path(data_dir)).fingerprint(),
    }


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(recorded: dict, current: dict) -> list[str]:
    lines = []
    rf, cf = recorded.get("gate_files", {}), current.get("gate_files", {})
    for p in sorted(rf.keys() | cf.keys()):
        if p not in cf:
            lines.append(f"gate file removed: {p}")
        elif p not in rf:
            lines.append(f"gate file added: {p}")
        elif rf[p] != cf[p]:
            lines.append(f"gate file changed: {p}")
    for k in sorted(recorded.get("knobs", {}).keys() | current.get("knobs", {}).keys()):
        a, b = recorded.get("knobs", {}).get(k), current.get("knobs", {}).get(k)
        if a != b:
            lines.append(f"knob {k}: {a} -> {b}")
    if recorded.get("judge_model") != current.get("judge_model"):
        lines.append(f"judge_model: {recorded.get('judge_model')} -> {current.get('judge_model')}")
    rp, cp = _flatten(recorded.get("policy", {})), _flatten(current.get("policy", {}))
    for k in sorted(rp.keys() | cp.keys()):
        if rp.get(k) != cp.get(k):
            lines.append(f"policy {k}: {rp.get(k)} -> {cp.get(k)}")
    if recorded.get("reference_fingerprint") != current.get("reference_fingerprint"):
        lines.append("reference fingerprint changed")
    return lines


def record(path: Path, data_dir: Path, today: str, root: Path = Path(".")) -> dict:
    path = Path(path)
    if path.exists():
        raise FreezeError(f"{path} exists: a second freeze is a new file and a new tag")
    if not os.getenv("JUDGE_MODEL"):
        # Unset, it falls back to LLM_MODEL, and the record would pin a model nobody chose.
        raise FreezeError("JUDGE_MODEL is unset: set it explicitly before freezing")
    head = _git_lines(Path(root), "rev-parse", "HEAD")[0]
    out = {"frozen": today, "commit": head, **gate_state(data_dir, root)}
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def check(path: Path, data_dir: Path, root: Path = Path(".")) -> list[str]:
    path = Path(path)
    if not path.exists():
        return [f"no freeze record at {path}"]
    recorded = json.loads(path.read_text(encoding="utf-8"))
    return diff(recorded, gate_state(data_dir, root))


def require_frozen(path: Path | None = None, data_dir: Path | None = None,
                   root: Path = Path(".")) -> None:
    lines = check(path if path is not None else FROZEN_GATE,
                  data_dir if data_dir is not None else real_data_dir(), root)
    if lines:
        raise FreezeError("gate differs from its freeze record:\n" + "\n".join(lines))


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("record", "check"):
        print("usage: python -m eval.freeze_gate record|check", file=sys.stderr)
        return 2
    try:
        if argv[0] == "record":
            record(FROZEN_GATE, real_data_dir(), date.today().isoformat())
            print(f"recorded {FROZEN_GATE}")
            return 0
    except FreezeError as e:
        print(e, file=sys.stderr)
        return 1
    lines = check(FROZEN_GATE, real_data_dir())
    for line in lines:
        print(line)
    if lines:
        return 1
    print("freeze check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
