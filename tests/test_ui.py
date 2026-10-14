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


import importlib
import json
import re
from pathlib import Path

import pytest

from app import config, orchestrator
from app import ui
from app.ui import (audit_offset, load_texts, make_respond, make_runner, panel_rows,
                    read_entries_since, render_panel)


def entry(item_id="c1", stage="read", store="corpus", action="admit", trust=1.0, signals=None):
    return {"ts": "t", "item_id": item_id, "store": store, "stage": stage, "action": action,
            "trust": trust, "reason": "r", "signals": signals or []}


def sig(name, score, reason="why", span=None):
    return {"signal": name, "score": score, "reason": reason, "evidence_span": span}


def write_lines(path, rows, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")


# --- wiring -----------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_run_agentic(question, **kwargs):
        calls.append((question, kwargs))
        return {"answer": "A", "citations": [], "path_trace": []}

    monkeypatch.setattr(orchestrator, "run_agentic", fake_run_agentic)
    return calls


@pytest.fixture
def fake_wiring(monkeypatch, tmp_path):
    audit = type("A", (), {"path": tmp_path / "gate_audit.jsonl"})()
    gate = type("G", (), {"audit": audit})()
    store = object()
    seen = {}

    def fake_gate(data_dir, chat=None, reference=None):
        seen["gate_dir"] = Path(data_dir)
        seen["reference"] = reference
        return gate

    def fake_store(data_dir):
        seen["store_dir"] = Path(data_dir)
        return store

    monkeypatch.setattr(ui, "default_read_gate", fake_gate)
    monkeypatch.setattr(ui, "default_experience_store", fake_store)
    return gate, store, seen


def test_flags_off_runner_passes_no_gate_and_no_store(recorded, fake_wiring, monkeypatch):
    monkeypatch.setattr(config, "UI_READ_GATE", False)
    monkeypatch.setattr(config, "UI_EXPERIENCE_STORE", False)
    runner, audit_path = ui._wiring()
    runner("q")
    assert recorded == [("q", {})]
    assert audit_path is None
    assert fake_wiring[2] == {}


def test_read_gate_flag_adds_only_the_gate(recorded, fake_wiring, monkeypatch):
    gate, _, seen = fake_wiring
    monkeypatch.setattr(config, "UI_READ_GATE", True)
    monkeypatch.setattr(config, "UI_EXPERIENCE_STORE", False)
    monkeypatch.delenv("GATE_DIR", raising=False)
    runner, audit_path = ui._wiring()
    runner("q")
    assert recorded == [("q", {"read_gate": gate})]
    assert audit_path == gate.audit.path
    assert seen["gate_dir"] == Path("data") and seen["reference"] is None
    assert "store_dir" not in seen


def test_store_flag_adds_only_the_store(recorded, fake_wiring, monkeypatch):
    _, store, seen = fake_wiring
    monkeypatch.setattr(config, "UI_READ_GATE", False)
    monkeypatch.setattr(config, "UI_EXPERIENCE_STORE", True)
    monkeypatch.delenv("GATE_DIR", raising=False)
    runner, audit_path = ui._wiring()
    runner("q")
    assert recorded == [("q", {"store": store})]
    assert audit_path is None
    assert seen["store_dir"] == Path("data") and "gate_dir" not in seen


def test_make_runner_passes_exactly_its_collaborators(recorded):
    make_runner()("q")
    make_runner(read_gate="g", store="s")("q2")
    assert recorded == [("q", {}), ("q2", {"read_gate": "g", "store": "s"})]


def test_flag_values_other_than_1_are_off(monkeypatch):
    try:
        for value in ("0", "false", "", "true", "yes", " 1"):
            monkeypatch.setenv("UI_READ_GATE", value)
            monkeypatch.setenv("UI_EXPERIENCE_STORE", value)
            importlib.reload(config)
            assert config.UI_READ_GATE is False and config.UI_EXPERIENCE_STORE is False, value
        monkeypatch.setenv("UI_READ_GATE", "1")
        monkeypatch.setenv("UI_EXPERIENCE_STORE", "1")
        importlib.reload(config)
        assert config.UI_READ_GATE is True and config.UI_EXPERIENCE_STORE is True
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# --- audit reading ----------------------------------------------------------

def test_audit_offset_of_absent_file_is_zero(tmp_path):
    assert audit_offset(tmp_path / "missing.jsonl") == 0
    assert read_entries_since(tmp_path / "missing.jsonl", 0) == []


def test_entries_before_offset_are_not_returned(tmp_path):
    p = tmp_path / "a.jsonl"
    write_lines(p, [entry("old")])
    off = audit_offset(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry("new")) + "\n")
    assert [e["item_id"] for e in read_entries_since(p, off)] == ["new"]


def test_write_stage_entries_are_not_returned(tmp_path):
    p = tmp_path / "a.jsonl"
    write_lines(p, [entry("r1"), entry("w1", stage="write"), entry("r2")])
    assert [e["item_id"] for e in read_entries_since(p, 0)] == ["r1", "r2"]


def test_offset_past_end_reads_from_start(tmp_path):
    p = tmp_path / "a.jsonl"
    write_lines(p, [entry("only")])
    assert [e["item_id"] for e in read_entries_since(p, 10_000)] == ["only"]


def test_partial_trailing_line_is_skipped(tmp_path):
    p = tmp_path / "a.jsonl"
    write_lines(p, [entry("whole")], tail='{"item_id": "half", "stage": "re')
    assert [e["item_id"] for e in read_entries_since(p, 0)] == ["whole"]


def test_load_texts_joins_chunks_and_experience_screened_text(tmp_path):
    chunks, exps = tmp_path / "chunks.jsonl", tmp_path / "experiences.jsonl"
    write_lines(chunks, [{"id": "c1", "text": "chunk text", "trusted": True}])
    write_lines(exps, [{"id": "e1", "task": "t", "approach": "a", "outcome": "o",
                        "success": True, "ts": "2026-01-01T00:00:00+00:00"}])
    texts = load_texts(chunks, exps)
    assert texts["c1"] == "chunk text"
    assert texts["e1"] == "task: t\napproach: a\noutcome: o"
    assert load_texts(tmp_path / "none.jsonl", tmp_path / "none2.jsonl") == {}


# --- rows -------------------------------------------------------------------

def test_top_signal_is_highest_score():
    rows = panel_rows([entry("c1", signals=[sig("structural", 0.2, "s1"),
                                             sig("judge", 0.9, "s2", "bad")])],
                      {"c1": "x bad y"})
    r = rows[0]
    assert (r["top_signal"], r["top_reason"], r["evidence_span"]) == ("judge", "s2", "bad")
    assert panel_rows([entry("c2")], {})[0]["top_signal"] is None


def test_verbatim_span_is_marked():
    rows = panel_rows([entry("c1", signals=[sig("judge", 0.9, span="bad bit")])],
                      {"c1": "a bad bit here"})
    assert rows[0]["span_verbatim"] is True
    assert "a <mark>bad bit</mark> here" in render_panel(rows, True)


def test_non_verbatim_span_is_shown_beside_text():
    rows = panel_rows([entry("c1", signals=[sig("structural", 0.8, span="decoded")])],
                      {"c1": "b64 blob"})
    assert rows[0]["span_verbatim"] is False
    out = render_panel(rows, True)
    assert "b64 blob" in out and "decoded" in out
    assert "b64 blob<mark>" not in out


def test_missing_text_shows_id():
    rows = panel_rows([entry("ghost-id")], {})
    assert rows[0]["text"] is None
    assert "ghost-id" in render_panel(rows, True)


def test_render_escapes_candidate_html():
    text = '<div style="display:none">x</div> <!-- hidden -->'
    rows = panel_rows([entry("c1", signals=[sig("judge", 0.9, "<b>r</b>", "<script>")])],
                      {"c1": text})
    out = render_panel(rows, True)
    assert "&lt;div style=&quot;display:none&quot;&gt;" in out
    assert "&lt;!-- hidden --&gt;" in out
    assert "&lt;b&gt;r&lt;/b&gt;" in out and "&lt;script&gt;" in out
    assert "<!--" not in out and "<script" not in out and "<b>" not in out
    allowed = {"div", "table", "thead", "tbody", "tr", "th", "td", "mark", "span", "p", "code"}
    assert {m.lower() for m in re.findall(r"</?([a-zA-Z][a-zA-Z0-9]*)", out)} <= allowed
    assert 'style="display:none"' not in out


def test_gate_off_renders_off_message_not_empty_table():
    out = render_panel([], False)
    assert "off" in out.lower() and "<table" not in out


def test_gate_on_with_no_rows_says_nothing_screened():
    out = render_panel([], True)
    assert "nothing was screened" in out.lower() and "<table" not in out


# --- respond ----------------------------------------------------------------

def test_respond_shows_only_this_querys_entries(tmp_path):
    audit = tmp_path / "gate_audit.jsonl"
    write_lines(audit, [entry("earlier-query")])

    def runner(q):
        with audit.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry("now-1")) + "\n")
            f.write(json.dumps(entry("now-w", stage="write")) + "\n")
            f.write(json.dumps(entry("now-2")) + "\n")
        return {"answer": "A", "citations": [], "path_trace": []}

    respond = make_respond(runner, audit, lambda: {"now-1": "first", "now-2": "second"})
    history, _, panel, box = respond("q", [])
    assert history[-1] == {"role": "assistant", "content": "A"} and box == ""
    assert "first" in panel and "second" in panel
    assert "earlier-query" not in panel and "now-w" not in panel


def test_respond_with_gate_off_says_off():
    respond = make_respond(lambda q: {"answer": "A", "citations": [], "path_trace": []}, None,
                           lambda: {})
    _, _, panel, _ = respond("q", [])
    assert "off" in panel.lower()
