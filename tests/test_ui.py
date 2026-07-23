from app.ui import format_response, build_demo
import gradio as gr

def test_badge_direct():
    ans, side = format_response({"answer": "A", "citations": ["[T, chunk#0]"],
                                 "path_trace": ["retrieve", "grade:sufficient", "generate"]})
    assert ans == "A"
    assert "✅ direct" in side
    assert "[T, chunk#0]" in side

def test_badge_retried():
    _, side = format_response({"answer": "A", "citations": [],
                               "path_trace": ["retrieve", "grade:weak", "rewrite",
                                              "retrieve", "grade:sufficient", "generate"]})
    assert "🔁 retrieval retried" in side

def test_build_demo_returns_blocks():
    demo = build_demo(runner=lambda q: {"answer": "A", "citations": [],
                                        "path_trace": ["generate"]})
    assert isinstance(demo, gr.Blocks)
