"""Read-gate overhead: LLM calls and wall-clock per question, gate off vs on.

Three arms over the same questions: `off` (no gate), `on-cold` (gate on an
empty verdict cache, so every chunk reaches the judge) and `on-warm` (a second
pass over the same cache). Makes no RAGAS calls.

    OVERHEAD_N=5 LLM_MIN_INTERVAL=2 python -m eval.overhead   # spends real LLM budget

The judge's calls are not counted separately: the `off` arm's calls per question
is the baseline, and judge calls are the difference.
"""
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from app import config

# Rewritten after every arm, so a quota error mid-run leaves the finished arms
# on disk. Outside any directory this script wipes.
RESULTS_DIR = Path("data/phase3_overhead")


@contextmanager
def count_calls(llm_module):
    """Count `llm_module.chat` calls. Patches the module attribute, so it sees
    the judge's calls only because the judge resolves `app.llm.chat` at call
    time rather than binding it at construction."""
    counter = Counter()
    original = llm_module.chat

    def counted(*args, **kwargs):
        counter["chat"] += 1
        return original(*args, **kwargs)

    llm_module.chat = counted
    try:
        yield counter
    finally:
        llm_module.chat = original


def measure(questions, arm_runner, llm_module=None):
    if llm_module is None:
        from app import llm as llm_module
    start = time.perf_counter()
    with count_calls(llm_module) as counter:
        for q in questions:
            arm_runner(q)
    total = time.perf_counter() - start
    n = len(questions)
    return {"questions": n, "llm_calls": counter["chat"], "total_seconds": total,
            "calls_per_question": counter["chat"] / n, "seconds_per_question": total / n}


def run_arms(questions, run_agentic, gate_factory, llm_module=None, on_arm=None):
    """`run_agentic(question, read_gate=)` and `gate_factory(data_dir)` are
    injected. `on_arm(arms_so_far)` runs after each arm so the caller can persist."""
    arms = {}

    def finish(name, read_gate):
        arms[name] = measure(questions, lambda q: run_agentic(q, read_gate=read_gate), llm_module)
        if on_arm is not None:
            on_arm(arms)

    finish("off", None)
    # Fresh temp dir, never data/: on-cold must start from an empty cache, and
    # the shared production cache must not be polluted by a measurement.
    data_dir = tempfile.mkdtemp(prefix="overhead-")
    try:
        finish("on-cold", gate_factory(data_dir))
        if arms["on-cold"]["llm_calls"] <= arms["off"]["llm_calls"]:
            raise SystemExit(
                f"on-cold made {arms['on-cold']['llm_calls']} LLM calls vs {arms['off']['llm_calls']} "
                "for off: judge calls read 0. app.llm.chat is not being resolved at call time; "
                "fix that before reporting any overhead number.")
        # Built after the cold pass: VerdictCache loads its file on open, so a
        # gate built earlier would not see what the cold pass wrote.
        finish("on-warm", gate_factory(data_dir))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return arms


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main(run_agentic=None, gate_factory=None, llm_module=None, questions=None) -> int:
    if run_agentic is None:
        from app.orchestrator import run_agentic as agentic
        from app.retrieval import Retriever
        from gate.read import default_read_gate

        retriever = Retriever()  # built once; every arm retrieves from the same index
        run_agentic = lambda q, read_gate=None: agentic(q, retriever=retriever, read_gate=read_gate)
        gate_factory = default_read_gate
    if questions is None:
        questions = [qa["question"] for qa in json.load(open("eval/qa_set.json", encoding="utf-8"))]

    # Pacing dominates seconds-per-question, so the value in force is recorded.
    config.LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "2"))
    n = int(os.getenv("OVERHEAD_N", "5"))
    results = {"model": config.LLM_MODEL, "llm_min_interval": config.LLM_MIN_INTERVAL,
               "overhead_n": n, "complete": False, "arms": {}}

    def on_arm(arms):
        results["arms"] = dict(arms)
        results["complete"] = len(arms) == 3
        save_results(results)

    run_arms(questions[:n], run_agentic, gate_factory, llm_module, on_arm)
    # Default ensure_ascii: a cp1252 Windows console must not crash after the
    # file is written.
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
