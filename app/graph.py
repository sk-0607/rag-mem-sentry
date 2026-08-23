from typing import TypedDict

from langgraph.graph import StateGraph, END

from app import llm as _llm
from app import websearch as _websearch
from app import config

REFUSAL = "Not grounded in the corpus."


class RAGState(TypedDict):
    query: str
    original_query: str
    docs: list
    grade: str
    retries: int
    web_tried: bool
    answer: str
    citations: list
    path_trace: list
    experiences: str


def build_graph(retriever, llm_module=_llm, web_module=_websearch):
    def retrieve_node(state):
        docs = retriever.retrieve(state["query"])
        for d in docs:
            d["source"] = "corpus"
        return {"docs": docs, "path_trace": state["path_trace"] + ["retrieve"]}

    def grade_node(state):
        g = "weak" if not state["docs"] else llm_module.grade(state["query"], state["docs"])
        return {"grade": g, "path_trace": state["path_trace"] + [f"grade:{g}"]}

    def rewrite_node(state):
        nq = llm_module.rewrite(state["original_query"])
        return {"query": nq, "retries": state["retries"] + 1,
                "path_trace": state["path_trace"] + ["rewrite"]}

    def web_node(state):
        docs = web_module.web_search(state["query"])
        for d in docs:
            d["source"] = "web"
        return {"docs": docs, "web_tried": True,
                "path_trace": state["path_trace"] + ["web_fallback"]}

    def generate_node(state):
        out = llm_module.answer(state["query"], state["docs"], state.get("experiences", ""))
        return {"answer": out["answer"], "citations": out["citations"],
                "path_trace": state["path_trace"] + ["generate"]}

    def refuse_node(state):
        return {"answer": REFUSAL, "citations": [],
                "path_trace": state["path_trace"] + ["refuse"]}

    def route(state):
        if state["grade"] == "sufficient":
            return "generate"
        if state["retries"] < config.MAX_RETRIES:
            return "rewrite"
        if not state["web_tried"] and config.WEB_ENABLED:
            return "web_fallback"
        return "refuse"

    g = StateGraph(RAGState)
    for name, fn in [("retrieve", retrieve_node), ("grade", grade_node),
                     ("rewrite", rewrite_node), ("web_fallback", web_node),
                     ("generate", generate_node), ("refuse", refuse_node)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {
        "generate": "generate", "rewrite": "rewrite",
        "web_fallback": "web_fallback", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_fallback", "grade")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()
