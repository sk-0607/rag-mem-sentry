from typing import TypedDict

from langgraph.graph import StateGraph, END

from app import llm as _llm
from gate.cache import content_hash
from gate.types import Action, Item


class RAGState(TypedDict):
    query: str
    docs: list
    answer: str
    citations: list
    path_trace: list
    experiences: str
    gate_excluded: list


def build_graph(retriever, llm_module=_llm, read_gate=None):
    def retrieve_node(state):
        docs = retriever.retrieve(state["query"])
        return {"docs": docs, "path_trace": state["path_trace"] + ["retrieve"]}

    def gate_node(state):
        docs = state["docs"]
        # No pre-existing id on a corpus chunk (fixtures and the class-1 trial's
        # synthetic chunks carry none) -> derive one from content so the gate
        # still has something stable to key its cache/audit on.
        items = [
            Item(id=doc["id"] if "id" in doc else content_hash(doc["text"])[:16],
                 text=doc["text"], kind="corpus",
                 # `is True`: a stored "true" string must not count as trusted.
                 metadata={"trusted": doc.get("trusted") is True})
            for doc in docs
        ]
        result = read_gate.screen_batch(items, "corpus")
        admitted, excluded = [], []
        for doc, item, verdict in zip(docs, items, result.verdicts):
            if verdict.action is Action.ADMIT:
                admitted.append(doc)
            else:
                excluded.append(item.id)
        trace = state["path_trace"] + [f"gate:{len(admitted)}/{len(docs)}"]
        if result.reduced_assurance:
            trace.append("gate:reduced_assurance")
        return {"docs": admitted, "gate_excluded": excluded, "path_trace": trace}

    def generate_node(state):
        out = llm_module.answer(state["query"], state["docs"], state.get("experiences", ""))
        return {"answer": out["answer"], "citations": out["citations"],
                "path_trace": state["path_trace"] + ["generate"]}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    if read_gate is None:
        g.add_edge("retrieve", "generate")
    else:
        g.add_node("gate", gate_node)
        g.add_edge("retrieve", "gate")
        g.add_edge("gate", "generate")
    g.add_edge("generate", END)
    return g.compile()
