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

def test_build_demo_returns_blocks():
    demo = build_demo(runner=lambda q: {"answer": "A", "citations": [],
                                        "path_trace": ["sq0:generate"]})
    assert isinstance(demo, gr.Blocks)
