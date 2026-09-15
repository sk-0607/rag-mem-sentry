from app.ui import format_response, build_demo
import gradio as gr

def test_format_response_lists_citations():
    ans, side = format_response({"answer": "A", "citations": ["[T, chunk#0]"],
                                 "path_trace": ["sq0:retrieve", "sq0:generate"]})
    assert ans == "A"
    assert "[T, chunk#0]" in side

def test_format_response_no_citations():
    _, side = format_response({"answer": "A", "citations": [], "path_trace": []})
    assert "_none_" in side

def test_build_demo_returns_blocks():
    demo = build_demo(runner=lambda q: {"answer": "A", "citations": [],
                                        "path_trace": ["sq0:generate"]})
    assert isinstance(demo, gr.Blocks)
