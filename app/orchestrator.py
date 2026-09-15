from pathlib import Path

from app import config
from app import llm as _llm
from app.graph import build_graph as _build_graph
from memory.gate import default_experience_gate
from memory.types import record_from_run


def format_experiences(records):
    """Render past ExperienceRecords into a labelled block for the prompt.

    Empty string on no records — it must add nothing to the prompt when
    there is no memory, not an empty "Past experiences:" header.
    """
    if not records:
        return ""
    blocks = [f"- task: {r.task}\n  approach: {r.approach}\n  outcome: {r.outcome}"
              for r in records]
    return "Past experiences:\n" + "\n".join(blocks)


def run_agentic(question, retriever=None, llm_module=_llm, graph_builder=_build_graph,
                store=None, gate=None):
    if retriever is None:
        from app.retrieval import Retriever
        retriever = Retriever()
    app = graph_builder(retriever, llm_module)

    records = store.retrieve(question, k=config.EXPERIENCE_K) if store is not None else []
    experiences = format_experiences(records)

    subqs = llm_module.decompose(question)
    sub_results, trace, contexts = [], [], []
    for i, sq in enumerate(subqs):
        state = app.invoke({
            "query": sq, "docs": [], "answer": "", "citations": [],
            "path_trace": [], "experiences": experiences,
        })
        ctx = [d["text"] for d in state["docs"]]
        sub_results.append({
            "sub_query": sq, "answer": state["answer"],
            "citations": state["citations"], "contexts": ctx,
            "path_trace": state["path_trace"],
        })
        trace += [f"sq{i}:{s}" for s in state["path_trace"]]
        contexts += ctx

    final = llm_module.synthesize(question, sub_results)

    if store is not None:
        # Write-time gating only (Task 15's ExperienceStore.write(gate=...));
        # the retrieve above is deliberately ungated — Task 20 needs that
        # undefended read path to demonstrate the attack it defends against.
        record = record_from_run(
            task=question,
            approach=", ".join(trace),
            outcome=final["answer"],
            # No refusal path any more, so every run produces an answer; there is
            # no failure signal left to derive success from.
            success=True,
        )
        if gate is None:
            # Derived from config at call time, not a hardcoded Path("data"):
            # a CWD-relative literal writes the gate's audit/cache/quarantine
            # wherever the process was started. Same fix Task 19 applied to
            # ingest/run_ingest.py via config.CHUNKS_PATH.
            gate = default_experience_gate(Path(config.EXPERIENCES_PATH).parent)
        store.write(record, gate=gate)

    return {"answer": final["answer"], "citations": final["citations"],
            "path_trace": trace, "contexts": contexts,
            "experiences": [r.id for r in records]}
