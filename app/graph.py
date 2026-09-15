from typing import TypedDict

from langgraph.graph import StateGraph, END

from app import llm as _llm


class RAGState(TypedDict):
    query: str
    docs: list
    answer: str
    citations: list
    path_trace: list
    experiences: str


def build_graph(retriever, llm_module=_llm):
    def retrieve_node(state):
        docs = retriever.retrieve(state["query"])
        return {"docs": docs, "path_trace": state["path_trace"] + ["retrieve"]}

    def generate_node(state):
        out = llm_module.answer(state["query"], state["docs"], state.get("experiences", ""))
        return {"answer": out["answer"], "citations": out["citations"],
                "path_trace": state["path_trace"] + ["generate"]}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
