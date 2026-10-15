"""Local web UI: the same runner and trust panel data as the Gradio UI, as JSON plus one page.

Not app/api.py: the panel exposes per-signal scores and reasons, which the Phase 6 spec keeps
off the public API. This server binds 127.0.0.1 only, like the Gradio demo it sits beside.
"""
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from app import config
from app.ui import _wiring, audit_offset, load_texts, panel_rows, read_entries_since

PAGE = Path(__file__).parent / "static" / "index.html"


class Ask(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question is blank")
        return v.strip()


def create_app(runner, audit_path, texts) -> FastAPI:
    gate_on = audit_path is not None
    # Gradio queues one query at a time; FastAPI runs sync endpoints in parallel threads.
    # The panel reads "audit entries since offset", so a second query must wait.
    lock = threading.Lock()
    app = FastAPI(title="RAG Mem Sentry")

    @app.get("/")
    def index():
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/status")
    def status():
        return {"gate_on": gate_on, "memory_on": config.UI_EXPERIENCE_STORE}

    @app.post("/ask")
    def ask(q: Ask):
        with lock:
            offset = audit_offset(audit_path) if gate_on else 0
            result = runner(q.question)
            rows = panel_rows(read_entries_since(audit_path, offset), texts()) if gate_on else []
        trace = result.get("path_trace", [])
        return {"answer": result["answer"], "citations": result["citations"],
                "gate_on": gate_on, "rows": rows,
                "reduced_assurance": any(t.endswith("gate:reduced_assurance") for t in trace),
                "experiences_excluded": result.get("experiences_excluded", [])}

    return app


def main():
    runner, audit_path = _wiring()
    app = create_app(runner, audit_path,
                     lambda: load_texts(Path(config.CHUNKS_PATH), Path(config.EXPERIENCES_PATH)))
    uvicorn.run(app, host="127.0.0.1", port=7861)


if __name__ == "__main__":
    main()
