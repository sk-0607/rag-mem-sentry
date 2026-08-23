"""Reproduce Task 20's experience-store calibration run.

Generates the clean experience set that `eval/experience_fpr.py` then measures.
Unlike the corpus, there is no ready-made clean set here — it only exists once
the real agent has run, so this script is the experience store's equivalent of
`python -m ingest.run_ingest`.

**This spends real LLM budget.** Each question fans out into several sub-queries,
so 15 questions is on the order of 150 generation calls; the Task 20 run took
498.5s paced at `LLM_MIN_INTERVAL=4`. It is committed so the measurement is
reproducible, not because it is cheap to re-run.

    CALIBRATION_N=15 python -m eval.experience_calibration
    python -m eval.experience_fpr data/gate_audit.jsonl

Appends to the live store rather than starting clean: reproducing the
measurement should not mean discarding real records that already cost budget.
`experience_fpr`'s counts are over everything in the audit log, so a re-run
raises `total`, which is why the README quotes the count it was measured at.
"""
import json
import os
from pathlib import Path

from app import config


def calibrate(questions, runner, store) -> list[dict]:
    """Run each question through `runner` with `store` attached, in order.

    Sequential on purpose, and not just for rate limits: each run writes its
    own experience record, so run N+1 retrieves against a store that run N
    changed. Parallelising would make the set non-reproducible.
    """
    rows = []
    for i, question in enumerate(questions):
        result = runner(question, store=store)
        rows.append({
            "i": i,
            "question": question,
            "answer": result["answer"],
            "path_trace": result["path_trace"],
            "experiences": result["experiences"],
        })
    return rows


def main() -> int:
    from app.orchestrator import run_agentic
    from app.retrieval import Retriever
    from memory.store import default_experience_store

    qa = json.load(open("eval/qa_set.json", encoding="utf-8"))
    n = int(os.getenv("CALIBRATION_N", "15"))
    questions = [q["question"] for q in qa[:n]]

    # Same free-tier pacing eval/run_eval.py uses; the answer-generation phase
    # is the bursty one.
    config.LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4"))

    retriever = Retriever()
    store = default_experience_store(Path(config.EXPERIENCES_PATH).parent)

    def runner(question, store):
        return run_agentic(question, retriever=retriever, store=store)

    rows = calibrate(questions, runner, store)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
