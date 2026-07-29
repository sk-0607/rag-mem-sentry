import types
from app.orchestrator import run_agentic

class FakeApp:
    def __init__(self): self.invocations = []
    def invoke(self, state):
        self.invocations.append(state["query"])
        return {**state, "docs": [{"text": f"ctx-{state['query']}", "source": "corpus"}],
                "answer": f"ans-{state['query']}",
                "citations": [f"[{state['query']}, chunk#0]"],
                "path_trace": ["retrieve", "grade:sufficient", "generate"]}

def make_llm(subqs):
    m = types.SimpleNamespace()
    m.decompose = lambda q, client=None: subqs
    m.synthesize = lambda q, sr, client=None: {
        "answer": " + ".join(r["answer"] for r in sr),
        "citations": [c for r in sr for c in r["citations"]]}
    return m

def test_runs_each_subquery_sequentially_and_merges():
    fake_app = FakeApp()
    out = run_agentic("compound?",
                      retriever=object(),
                      llm_module=make_llm(["sq1", "sq2"]),
                      graph_builder=lambda r: fake_app)
    assert fake_app.invocations == ["sq1", "sq2"]          # sequential, both run
    assert out["answer"] == "ans-sq1 + ans-sq2"
    assert out["citations"] == ["[sq1, chunk#0]", "[sq2, chunk#0]"]
    assert any(s.startswith("sq0:") for s in out["path_trace"])
    assert any(s.startswith("sq1:") for s in out["path_trace"])
    assert out["contexts"] == ["ctx-sq1", "ctx-sq2"]

def test_single_subquery_passthrough():
    fake_app = FakeApp()
    llm = make_llm(["only"])
    llm.synthesize = lambda q, sr, client=None: {"answer": sr[0]["answer"],
                                                 "citations": sr[0]["citations"]}
    out = run_agentic("atomic?", retriever=object(), llm_module=llm,
                      graph_builder=lambda r: fake_app)
    assert out["answer"] == "ans-only"
