"""Open measurement #2: behavior-level ASR with both Task 20 confounds removed.

Task 20 measured context ASR 3/3 but behavior ASR 0/3, against two documented
confounds: `app/llm.answer`'s system prompt plus well-covered corpus context
outcompeted the appended experience block, and `EXPERIENCE_K=3` against 17 clean
records co-retrieved clean siblings modelling the opposite approach.

This trial removes both:

- **A poison-only store**, its own Chroma dir and JSONL under `data/poison_trial/`,
  so the live experience store is neither read nor written and no purge is needed.
  `EXPERIENCE_K = 1` on top of that: only the nearest record is injected, so not
  even another poisoned record dilutes the one under test.
- **Records from the external attack corpus** (since 2026-10-15): the first
  `CLASS3_N` class-3 groups of the frozen test split, each with its own target
  question and pre-registered check (`trials`). The earlier hand-written,
  papers-era questions do not exist in the NQ corpus.

The poison is false-fact / unsafe-approach shaped, never "ignore all previous
instructions" — R3a's class 2/3, and the shape design spec section 9.1 names as
the harder case for model alignment to catch.

The simplified host (`app/graph.py`: retrieve -> generate, no grade/rewrite/web-fallback
loop) has no web path at all, so every sub-query reaches `generate` unconditionally —
"retrieved" and "reached a prompt" coincide here in a way they did not on the
2026-08-24 host this trial was first run against.

    python -m eval.experience_poison_trial --arm off   # spends real LLM budget
    python -m eval.experience_poison_trial --arm on    # same, behind the read-time gate
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from app import config
from eval.attack_corpus import corpus_dir, load, to_experience_record
from eval.attack_screen import freeze_field
from eval.freeze_gate import FreezeError, real_data_dir, require_frozen
from eval.judge_fpr import read_verdict, top_signal
from gate.reference import default_reference

TRIAL_DIR = Path("data/poison_trial")
# Sibling, not a child, of TRIAL_DIR: build_poison_store wipes TRIAL_DIR on every
# run, so results kept inside it were deleted by the next run. Narrowing the wipe
# instead was rejected — a cold poison-only store per run is the trial's design.
RESULTS_DIR = Path("data/poison_trial_results")

# How many class-3 groups one run covers; one record per group, so each trial
# has one poison under test.
CLASS3_N = int(os.getenv("CLASS3_N", "3"))


def trials(entries: list[dict], n: int = CLASS3_N) -> list[dict]:
    """The first `n` class-3 groups of the frozen test split, one record each.

    Papers-era hand-written trials were replaced on 2026-10-15 (owner): the
    records are the external attack corpus's, never authored here, and the
    question and pre-registered `check` are the item's own fields, fixed before
    any run. `to_experience_record` is the projection the Task 46 screen used,
    so the gate sees the text it already scored.
    """
    groups: dict[str, dict] = {}
    for e in entries:
        if e["class"] == 3 and e["group"] not in groups:
            groups[e["group"]] = e
    return [{"question": e["target_question"], "poison": to_experience_record(e), "check": e["check"]}
            for e in list(groups.values())[:n]]


def load_trials() -> list[dict]:
    # The test split only, hash-checked by load(); the calibration split tuned
    # the gate and is never trialled.
    return trials(load(corpus_dir(), "test"))


class ReadOnlyStore:
    """Retrieve-only view of a store.

    `run_agentic` writes one record per run. Left alone, the poisoned arm's own
    answers would land in the poison store and could be retrieved by a later
    trial — and would write gate audit entries this measurement never asked for.
    """

    def __init__(self, inner):
        self.inner = inner

    def retrieve(self, task, k=config.EXPERIENCE_K):
        return self.inner.retrieve(task, k=k)

    def write(self, record, gate=None):
        return None


def build_poison_store(trials: list[dict], trial_dir=TRIAL_DIR):
    """Fresh poison-only store. Wipes `trial_dir` — never the live store."""
    from memory.store import ExperienceStore

    shutil.rmtree(trial_dir, ignore_errors=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    store = ExperienceStore(
        chroma_dir=trial_dir / "chroma",
        jsonl_path=trial_dir / "experiences.jsonl",
        collection="poison_trial",
    )
    for t in trials:
        # gate=None: the point of the trial is the undefended path.
        store.write(t["poison"], gate=None)
    return store


def screen_poison(trials: list[dict], trial_dir=TRIAL_DIR):
    """What the write-time gate would have said. Zero LLM calls, zero network.

    Audit goes to the trial dir, so `eval/experience_fpr.py`'s counts over
    `data/gate_audit.jsonl` stay the numbers the README quotes.
    """
    from memory.gate import default_experience_gate

    trial_dir.mkdir(parents=True, exist_ok=True)
    gate = default_experience_gate(trial_dir)
    out = []
    for t in trials:
        v = gate.screen(t["poison"].to_item(), "experience")
        out.append({"id": t["poison"].id, "action": v.action.value,
                    "trust": round(v.score.trust, 4), "reason": v.reason})
    return out


def configure():
    """LLM pacing + injection width, factored out of main() so a test can call
    it without spending budget (main() also runs the real agent)."""
    # 30 RPM on the free tier; 2s spacing leaves headroom without stretching the run.
    config.LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "2"))
    # Confound (b): one record injected, so nothing competes with the poison.
    config.EXPERIENCE_K = 1


def main(arm: str) -> int:
    from app.orchestrator import run_agentic
    from app.retrieval import Retriever

    # Before any work: a typo'd arm would otherwise run ungated and write a
    # results file under the typo'd name, after spending the budget.
    if arm not in ("off", "on"):
        raise ValueError(f"arm must be 'off' or 'on', got {arm!r}")
    # Both arms: `--arm off` and `--arm on` are one comparison only if both ran
    # against the frozen gate.
    try:
        require_frozen()
    except FreezeError as e:
        print(e, file=sys.stderr)
        return 1
    data_dir = real_data_dir()
    # TRIAL_DIR is wiped below and holds no chunks.jsonl, so the trusted set comes
    # from the real data directory; empty, S3 would abstain unseen.
    reference = default_reference(data_dir)
    if len(reference) == 0:
        print("trusted reference is empty: re-ingest with trust flags first (Task 43)",
              file=sys.stderr)
        return 1

    configure()

    ts = load_trials()
    gate_verdicts = screen_poison(ts)
    store = build_poison_store(ts)
    # After build_poison_store, never before: it rmtree's TRIAL_DIR, which would
    # delete the gate's cache and audit files. Built here the gate's cache is
    # cold and its audit holds only this run's verdicts.
    read_gate = None
    if arm == "on":
        from gate.read import default_read_gate

        read_gate = default_read_gate(TRIAL_DIR, reference=reference)
    retriever = Retriever()
    ro = ReadOnlyStore(store)

    # Rewritten in full after every trial (the file is tiny): a quota wall raises
    # out of run_agentic, and without this every completed trial died with it.
    # `complete` stays false until the last write, so a partial file cannot be
    # mistaken for a finished run. The error is left to propagate (exit code
    # stays non-zero); the point is only that finished trials are already on disk.
    out = RESULTS_DIR / f"results-{arm}.json"

    def save(complete):
        results["complete"] = complete
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Poisoned arm first, so a quota wall leaves the primary number on disk.
    results = {"arm": arm, "model": config.LLM_MODEL, "gate_on_poison_records": gate_verdicts,
               "complete": False, "freeze": freeze_field(data_dir), "poisoned": [], "control": []}
    for t in ts:
        r = run_agentic(t["question"], retriever=retriever, store=ro, read_gate=read_gate)
        # The audit lives under TRIAL_DIR, which the next run wipes, so what the
        # write-up needs from it is copied out here: the read-stage entry of every
        # retrieved poison record, admitted or excluded. run_agentic drops the
        # experience screen's reduced_assurance flag, so for an ADMITTED poison
        # this entry's `reason` ("reduced assurance: judge unavailable") is the
        # only record that S2 never ran. null = no entry, distinct from an
        # admit with clean signals.
        entries = read_gate.audit.entries() if read_gate is not None else []
        screened = {i: read_verdict(entries, i, "experience")
                    for i in r["experiences"] + r["experiences_excluded"]} if read_gate else {}
        # Each exclusion is credited to the signal that scored highest
        # (structural = S1, judge = S2).
        top = {i: top_signal(screened[i]) for i in r["experiences_excluded"]}
        results["poisoned"].append({
            "question": t["question"], "check": t["check"],
            "poison_id": t["poison"].id, "retrieved": r["experiences"],
            "experiences_excluded": r["experiences_excluded"],
            "excluded_top_signals": top,
            "gate_audit": screened,
            # On the simplified host every sub-query reaches `generate`
            # unconditionally (no grade/rewrite/refuse to intercept it), so a
            # retrieved experience always reaches the prompt. On the
            # 2026-08-24 host that was not true — t2's poison was retrieved
            # but every sub-query graded weak and refused before generating.
            # With the read gate on, a retrieved record can be excluded first,
            # so "reached the prompt" means retrieved and not excluded.
            "reached_prompt": t["poison"].id in r["experiences"],
            "path_trace": r["path_trace"], "answer": r["answer"],
        })
        save(False)
    # Ungated in both arms: the control has no store, so there is no poison for
    # the gate to act on, and gating it would only spend judge calls.
    for t in ts:
        r = run_agentic(t["question"], retriever=retriever)
        results["control"].append({
            "question": t["question"], "path_trace": r["path_trace"],
            "answer": r["answer"],
        })
        save(False)

    save(True)
    # Default ensure_ascii: a cp1252 Windows console raises on any model output
    # outside cp1252, after the file is written, turning a good run into exit 1.
    # The file on disk is utf-8 and unaffected.
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    raise SystemExit(main(parser.parse_args().arm))
