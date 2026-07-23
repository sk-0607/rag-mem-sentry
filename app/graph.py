from typing import TypedDict

from langgraph.graph import StateGraph, END

from app import llm as _llm
from app import config


class RAGState(TypedDict):
    query: str
    original_query: str
    docs: list
    grade: str
    retries: int
    answer: str
    citations: list
    path_trace: list


def build_graph(retriever, llm_module=_llm):
    def retrieve_node(state):
        docs = retriever.retrieve(state["query"])
        return {"docs": docs, "path_trace": state["path_trace"] + ["retrieve"]}

    def grade_node(state):
        if not state["docs"]:
            g = "weak"
        else:
            g = llm_module.grade(state["query"], state["docs"])
        return {"grade": g, "path_trace": state["path_trace"] + [f"grade:{g}"]}

    def rewrite_node(state):
        new_q = llm_module.rewrite(state["original_query"])
        return {"query": new_q, "retries": state["retries"] + 1,
                "path_trace": state["path_trace"] + ["rewrite"]}

    def generate_node(state):
        out = llm_module.answer(state["query"], state["docs"])
        return {"answer": out["answer"], "citations": out["citations"],
                "path_trace": state["path_trace"] + ["generate"]}

    def route(state):
        if state["grade"] == "sufficient" or state["retries"] >= config.MAX_RETRIES:
            return "generate"
        return "rewrite"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def run(question, retriever=None):
    if retriever is None:
        from app.retrieval import Retriever
        retriever = Retriever()
    app = build_graph(retriever)
    state = app.invoke({
        "query": question, "original_query": question, "docs": [],
        "grade": "", "retries": 0, "answer": "", "citations": [],
        "path_trace": [],
    })
    return {"answer": state["answer"], "citations": state["citations"],
            "path_trace": state["path_trace"],
            "contexts": [d["text"] for d in state["docs"]]}
