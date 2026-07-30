from app.ui import format_response, build_demo
import gradio as gr

def test_badge_direct():
    ans, side = format_response({"answer": "A", "citations": ["[T, chunk#0]"],
                                 "path_trace": ["sq0:retrieve", "sq0:grade:sufficient",
                                                "sq0:generate"]})
    assert ans == "A"
    assert "✅ direct" in side
    assert "[T, chunk#0]" in side

def test_badge_retried():
    _, side = format_response({"answer": "A", "citations": [],
                               "path_trace": ["sq0:retrieve", "sq0:grade:sufficient",
                                              "sq0:generate", "sq1:retrieve",
                                              "sq1:grade:weak", "sq1:rewrite",
                                              "sq1:retrieve", "sq1:grade:sufficient",
                                              "sq1:generate"]})
    assert "🔁 retrieval retried" in side

def test_badge_web_fallback_outranks_retry():
    # web_fallback is only reachable after the rewrite budget is spent, so both
    # nodes are in the trace; the badge must report the more informative one.
    _, side = format_response({"answer": "A", "citations": ["[web: owasp.org]"],
                               "path_trace": ["sq0:retrieve", "sq0:grade:weak",
                                              "sq0:rewrite", "sq0:retrieve",
                                              "sq0:grade:weak", "sq0:web_fallback",
                                              "sq0:grade:sufficient", "sq0:generate"]})
    assert "🌐 web fallback" in side
    assert "retrieval retried" not in side
    assert "[web: owasp.org]" in side

def test_build_demo_returns_blocks():
    demo = build_demo(runner=lambda q: {"answer": "A", "citations": [],
                                        "path_trace": ["sq0:generate"]})
    assert isinstance(demo, gr.Blocks)
