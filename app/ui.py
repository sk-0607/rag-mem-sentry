import html
import json
import os
from pathlib import Path

import gradio as gr

from app import config, orchestrator
from gate.read import default_read_gate
from memory.store import default_experience_store
from memory.types import ExperienceRecord


def format_response(result):
    cites = "\n".join(f"- {c}" for c in result["citations"]) or "_none_"
    side = f"**Citations:**\n{cites}"
    return result["answer"], side


# --- trust panel data: read from the audit log, never from graph state -------

def audit_offset(path) -> int:
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def read_entries_since(path, offset: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = p.read_bytes()
    # Shorter than the offset means data/ was wiped mid-query: read it all rather
    # than show an empty panel that reads as "nothing was screened".
    if offset > len(data):
        offset = 0
    out = []
    for line in data[offset:].decode("utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue  # a concurrent writer's half-written last line
        if isinstance(e, dict) and e.get("stage") == "read":
            out.append(e)
    return out


def _jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def load_texts(chunks_path, experiences_path) -> dict[str, str]:
    texts = {r["id"]: r["text"] for r in _jsonl(chunks_path) if "id" in r and "text" in r}
    for r in _jsonl(experiences_path):
        try:
            # screened_text(), not one field: the panel shows what the gate judged.
            texts[r["id"]] = ExperienceRecord(**r).screened_text()
        except (TypeError, KeyError):
            continue
    return texts


def panel_rows(entries: list[dict], texts: dict[str, str]) -> list[dict]:
    rows = []
    for e in entries:
        signals = e.get("signals") or []
        top = max(signals, key=lambda s: s["score"]) if signals else None
        text = texts.get(e.get("item_id"))
        span = top.get("evidence_span") if top else None
        rows.append({
            "store": e.get("store"), "item_id": e.get("item_id"), "action": e.get("action"),
            "trust": e.get("trust"),
            "top_signal": top["signal"] if top else None,
            "top_reason": top.get("reason") if top else None,
            "evidence_span": span, "text": text,
            "span_verbatim": bool(span) and text is not None and span in text,
        })
    return rows


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _text_cell(r: dict) -> str:
    text, span = r["text"], r["evidence_span"]
    if text is None:
        return f'<code>{_esc(r["item_id"])}</code>'
    # Escape the three parts, then join: marking up raw text first would let a
    # candidate's own tags through around the <mark>.
    if r["span_verbatim"]:
        i = text.index(span)
        body = f"{_esc(text[:i])}<mark>{_esc(span)}</mark>{_esc(text[i + len(span):])}"
    else:
        body = _esc(text)
    cell = f'<div class="tp-text">{body}</div>'
    if span and not r["span_verbatim"]:
        cell += f'<div class="tp-aside">evidence (not verbatim): <code>{_esc(span)}</code></div>'
    return cell


def render_panel(rows: list[dict], gate_on: bool) -> str:
    # Every candidate string goes through _esc: class-1 poison hides itself with
    # CSS and HTML comments, and would do so here if rendered raw.
    if not gate_on:
        return ('<p class="tp-note">Read gate is off (set <code>UI_READ_GATE=1</code>): '
                "retrieved passages and memories reach the model unscreened.</p>")
    if not rows:
        return '<p class="tp-note">Read gate on: nothing was screened for this query.</p>'
    body = []
    for r in rows:
        trust = "" if r["trust"] is None else f'{r["trust"]:.2f}'
        action = _esc(r["action"])
        signal = _esc(r["top_signal"]) if r["top_signal"] else "&mdash;"
        reason = f'<div class="tp-reason">{_esc(r["top_reason"])}</div>' if r["top_reason"] else ""
        body.append(
            "<tr>"
            f'<td><span class="tp-badge tp-{action}">{action}</span></td>'
            f'<td class="tp-num">{trust}</td>'
            f"<td>{signal}{reason}</td>"
            f'<td><span class="tp-store">{_esc(r["store"])}</span> <code>{_esc(r["item_id"])}</code>'
            f"{_text_cell(r)}</td>"
            "</tr>")
    return ('<div class="tp"><table><thead><tr><th>action</th><th>trust</th><th>top signal</th>'
            f'<th>candidate</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# --- wiring -----------------------------------------------------------------

def make_runner(read_gate=None, store=None):
    def run(question):
        # Only what is set: flags off must stay byte-for-byte run_agentic(question).
        kwargs = {}
        if read_gate is not None:
            kwargs["read_gate"] = read_gate
        if store is not None:
            kwargs["store"] = store
        return orchestrator.run_agentic(question, **kwargs)
    return run


def _wiring():
    # The data directory eval/prewarm.py fills, so the prewarmed cache serves the UI.
    data_dir = Path(os.getenv("GATE_DIR", "data"))
    read_gate = default_read_gate(data_dir) if config.UI_READ_GATE else None
    store = default_experience_store(data_dir) if config.UI_EXPERIENCE_STORE else None
    return make_runner(read_gate, store), (read_gate.audit.path if read_gate is not None else None)


def make_respond(runner, audit_path, texts):
    gate_on = audit_path is not None

    def respond(message, history):
        # Single user: a second query running at once would interleave its entries.
        offset = audit_offset(audit_path) if gate_on else 0
        result = runner(message)
        answer, side_md = format_response(result)
        rows = panel_rows(read_entries_since(audit_path, offset), texts()) if gate_on else []
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, side_md, render_panel(rows, gate_on), ""

    return respond


PANEL_CSS = """
.tp table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.tp th { text-align: left; font-weight: 600; padding: 6px 8px; border-bottom: 2px solid var(--border-color-primary); }
.tp td { vertical-align: top; padding: 8px; border-bottom: 1px solid var(--border-color-primary); }
.tp-num { font-variant-numeric: tabular-nums; }
.tp-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-weight: 600; font-size: 0.75rem; }
.tp-admit { background: #dcfce7; color: #166534; }
.tp-quarantine { background: #fef3c7; color: #92400e; }
.tp-block { background: #fee2e2; color: #991b1b; }
.tp-store { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7; }
.tp-reason, .tp-aside { font-size: 0.8rem; opacity: 0.8; margin-top: 2px; }
.tp-text { margin-top: 4px; max-height: 9rem; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
.tp mark { background: #fde68a; color: #1f2937; padding: 0 2px; border-radius: 2px; }
.tp-note { opacity: 0.8; font-style: italic; }
"""


def build_demo(runner=None, audit_path: Path | None = None, texts_loader=load_texts):
    if runner is None:
        runner, audit_path = _wiring()
    gate_on = audit_path is not None
    respond = make_respond(
        runner, audit_path,
        lambda: texts_loader(Path(config.CHUNKS_PATH), Path(config.EXPERIENCES_PATH)))
    with gr.Blocks(title="Agentic RAG") as demo:
        gr.Markdown("# Agentic RAG\nAsk about AI-security papers. "
                    "Answers are generated from a retrieved corpus.")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460)
                box = gr.Textbox(placeholder="e.g. What is an indirect prompt injection?",
                                 show_label=False)
            with gr.Column(scale=1):
                side = gr.Markdown("_Ask a question to see retrieval path + citations._")
        gr.Markdown("### Trust panel")
        panel = gr.HTML(render_panel([], False) if not gate_on else
                        '<p class="tp-note">Read gate on: ask a question to see what it screened.</p>',
                        # css_template, not Blocks(css=): Gradio 6 moved css to launch(),
                        # and this keeps the styling on the panel alone.
                        css_template=PANEL_CSS, elem_classes=["trust-panel"])

        box.submit(respond, [box, chatbot], [chatbot, side, panel, box])
    return demo
