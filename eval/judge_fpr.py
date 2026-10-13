"""R5: how often the S2 judge flags clean-but-attack-quoting corpus chunks.

Same argument as eval/fpr.py: the corpus holds no real attacks, so every
read-time non-admit is a false positive by construction. The set is the top-k
chunks the real pipeline would hand the gate for each QA question, plus the
chunks S1 already quarantined at ingest — the hardest quoted-attack content.
"""
import json
from collections import Counter
from pathlib import Path

from app import config
from eval.fpr import dedupe_by_item_id, false_positive_rate
from gate.audit import AuditLog
from gate.types import Item


def hard_negative_items(questions: list[str], retriever, held: list[Item]) -> list[Item]:
    # Raw questions, not run_agentic's decomposition: decomposing costs an LLM
    # call and the measurement is of the gate, not of the sub-query planner.
    # Item construction mirrors app/graph.py's gate_node so ids and text — and
    # therefore the cache key — are byte-identical to what the pipeline sees.
    items = [
        Item(id=doc["id"], text=doc["text"], kind="corpus")
        for q in questions
        for doc in retriever.retrieve(q, k=config.TOP_K)
    ]
    seen: dict[str, Item] = {}
    for item in items + held:
        seen.setdefault(item.id, item)
    return list(seen.values())


def top_signal(entry: dict) -> str:
    # "Top" is the highest-scoring signal in the entry: it names what drove a
    # flag, so S2-topped false positives can be told apart from S1-topped ones.
    # Shared with the attack trials, which credit each exclusion the same way.
    return max(entry["signals"], key=lambda s: s["score"])["signal"]


def read_verdict(entries: list[dict], item_id: str, store: str) -> dict | None:
    """The last read-stage audit entry for `item_id` in `store`, trimmed to what
    a write-up needs; None when there is none. The `reason` matters: it carries
    "reduced assurance: <signal> unavailable" when a signal failed, which is the
    only trace that an admit was reached without every signal having run."""
    match = [e for e in entries
             if e["item_id"] == item_id and e["store"] == store and e["stage"] == "read"]
    if not match:
        return None
    return {k: match[-1][k] for k in ("action", "trust", "reason", "signals")}


def summarize(audit_entries: list[dict]) -> dict:
    stats = false_positive_rate(audit_entries, stage="read")
    read = [e for e in audit_entries if e.get("store") == "corpus" and e.get("stage") == "read"]
    flagged = [e for e in read if e["action"] != "admit"]
    top = Counter(top_signal(e) for e in flagged)
    # Over every read entry, admits included: a judge that labels 40 chunks
    # `quotes` and admits them is the evidence the rubric separates the two.
    labels = Counter(s["reason"] for e in read for s in e["signals"] if s["signal"] == "judge")
    return {**stats, "by_top_signal": dict(top), "judge_labels": dict(labels)}


def main() -> int:
    from app.retrieval import Retriever
    from gate.cache import VerdictCache
    from gate.policy import DEFAULT_POLICY
    from gate.quarantine import Quarantine
    from gate.read import ReadGate
    from gate.signals.judge import JudgeSignal
    from gate.signals.structural import StructuralSignal

    data_dir = Path(config.CHUNKS_PATH).parent
    questions = [q["question"] for q in json.loads(Path("eval/qa_set.json").read_text(encoding="utf-8"))]
    items = hard_negative_items(
        questions, Retriever(), Quarantine(data_dir / "gate_quarantine.jsonl").held()
    )

    # default_read_gate's wiring with one difference: a separate audit log, so
    # this measurement reads only its own entries. The cache stays shared —
    # judged corpus verdicts are meant to be reused by later runs.
    audit = AuditLog(data_dir / "phase3_r5" / "gate_audit.jsonl")
    gate = ReadGate(
        signals=[StructuralSignal(), JudgeSignal()],
        policy=DEFAULT_POLICY,
        cache=VerdictCache(data_dir / "gate_cache.jsonl"),
        audit=audit,
        batch_max=config.JUDGE_BATCH_MAX,
    )
    gate.screen_batch(items, "corpus")

    # Deduped like eval/fpr.py: the audit file is append-only across reruns.
    print(json.dumps(summarize(dedupe_by_item_id(audit.entries())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
