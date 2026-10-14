"""Summarise the probe history, and count verdict flips against the final screen.

    python -m eval.probe_summary [log path]                     # default data/phase5_probe/probes.jsonl
    python -m eval.probe_summary --final data/phase5_screen/results.json

Offline, no LLM call. A flip is a final-set item whose last probe bit differs
from its final verdict: the verdict noise the probe bit carried (spec §10.2).
"""
import hashlib
import json
import sys
from pathlib import Path

from eval.attack_corpus import corpus_dir, load, to_gate_item

DEFAULT_LOG = Path("data/phase5_probe/probes.jsonl")


def read_log(path: Path) -> list[dict]:
    # utf-8-sig: a log copied through a Windows editor gains a BOM.
    text = Path(path).read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def summarise(rows: list[dict]) -> dict:
    sessions: dict[str, dict] = {}
    for r in rows:
        s = sessions.setdefault(str(r["session"]), {"counted": 0, "cached": 0, "rate_limited": 0,
                                                    "unavailable": 0, "calls": 0, "classes": {}})
        s["calls"] += r["calls"]
        c = s["classes"].setdefault(str(r["class"]), {"probed": 0, "admitted": 0})
        if r["status"] == 429:
            s["rate_limited"] += 1
        elif r["status"] == 503:
            s["unavailable"] += 1
        elif r["status"] == 200:
            s["counted"] += r["counted"]
            s["cached"] += r["cached"]
            c["probed"] += 1
            c["admitted"] += r["admitted"]
    for s in sessions.values():
        for c in s["classes"].values():
            c["rate"] = c["admitted"] / c["probed"] if c["probed"] else None

    ok = [r for r in rows if r["status"] == 200]
    ever: dict[str, set] = {}
    for r in ok:
        if r["admitted"]:
            ever.setdefault(str(r["class"]), set()).add(r["item_hash"])
    return {"sessions": sessions,
            "round": {"distinct_items": len({r["item_hash"] for r in ok}),
                      "ever_admitted": {c: len(hs) for c, hs in sorted(ever.items())}}}


def item_hash(entry: dict) -> str:
    item, _ = to_gate_item(entry)
    return hashlib.sha256(item.text.encode("utf-8")).hexdigest()


def verdict_flips(probe_rows: list[dict], entries: list[dict], screen_rows: list[dict]) -> dict:
    latest: dict[str, dict] = {}
    for r in probe_rows:
        if r["status"] == 200 and (r["item_hash"] not in latest or r["ts"] > latest[r["item_hash"]]["ts"]):
            latest[r["item_hash"]] = r
    final = {r["id"]: r["action"] == "admit" for r in screen_rows}
    out = {"final_items": len(entries), "probed_in_final": 0, "admitted_then_excluded": 0,
           "excluded_then_admitted": 0}
    for e in entries:
        probe = latest.get(item_hash(e))
        if probe is None:
            continue
        out["probed_in_final"] += 1
        if probe["admitted"] and not final[e["id"]]:
            out["admitted_then_excluded"] += 1
        elif not probe["admitted"] and final[e["id"]]:
            out["excluded_then_admitted"] += 1
    return out


def main(argv: list[str]) -> int:
    final = None
    if "--final" in argv:
        i = argv.index("--final")
        final = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    rows = read_log(Path(argv[0]) if argv else DEFAULT_LOG)
    out = summarise(rows)
    if final is not None:
        screen = json.loads(final.read_text(encoding="utf-8"))["rows"]
        out["flips"] = verdict_flips(rows, load(corpus_dir(), "adaptive"), screen)
    print(json.dumps(out, indent=2))
    for n, s in sorted(out["sessions"].items()):
        print(f"\nSession {n}\n\n| class | probed | admitted | rate |\n|---|---|---|---|")
        for c, v in sorted(s["classes"].items()):
            rate = "n/a" if v["rate"] is None else f"{v['rate']:.2f}"
            print(f"| {c} | {v['probed']} | {v['admitted']} | {rate} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
