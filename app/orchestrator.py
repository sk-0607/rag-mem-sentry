from app import llm as _llm
from app.graph import build_graph as _build_graph


def run_agentic(question, retriever=None, llm_module=_llm, graph_builder=_build_graph):
    if retriever is None:
        from app.retrieval import Retriever
        retriever = Retriever()
    app = graph_builder(retriever, llm_module)

    subqs = llm_module.decompose(question)
    sub_results, trace, contexts = [], [], []
    for i, sq in enumerate(subqs):
        state = app.invoke({
            "query": sq, "original_query": sq, "docs": [], "grade": "",
            "retries": 0, "web_tried": False, "answer": "", "citations": [],
            "path_trace": [],
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
    return {"answer": final["answer"], "citations": final["citations"],
            "path_trace": trace, "contexts": contexts}
