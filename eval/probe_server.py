"""The one-bit probe endpoint over the frozen read gate (Phase 5 spec §6).

    PROBE_SESSION=1 python -m eval.probe_server   # 127.0.0.1:8765, needs a clean freeze check

An attacker learns admitted / not admitted and nothing else: the action, trust,
signal scores and reasons never leave the server. Each session gets PROBE_CAP
uncached probes; a cached resubmission costs no LLM call, so it is not counted.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import llm
from eval.attack_corpus import to_gate_item
from eval.freeze_gate import FreezeError, real_data_dir, require_frozen
from gate.read import default_read_gate
from gate.reference import default_reference
from gate.types import Action

MAX_TEXT = 8000
PROBE_PORT = int(os.getenv("PROBE_PORT", "8765"))
PROBE_DIR = os.getenv("PROBE_DIR", "data/phase5_probe")
PROBE_CAP = int(os.getenv("PROBE_CAP", "20"))

_KEYS = {1: ("text",), 2: ("text",), 3: ("task", "approach", "outcome")}
_STORE = {1: "corpus", 2: "corpus", 3: "experience"}


class ProbeError(ValueError):
    pass


class CapReached(RuntimeError):
    pass


class CountingChat:
    """Wraps chat. At the cap it raises before the wrapped call, so the read
    gate sees an outage and caches nothing for the refused item."""

    def __init__(self, chat):
        self.chat = chat
        self.calls = 0
        self.refuse = False
        self.refused = False

    def reset(self):
        self.calls = 0
        self.refused = False

    def __call__(self, messages, client=None, model=None):
        if self.refuse:
            self.refused = True
            raise CapReached("probe cap reached")
        self.calls += 1
        return self.chat(messages, client=client, model=model)


def parse_probe(body) -> dict:
    if not isinstance(body, dict):
        raise ProbeError("body must be a JSON object")
    if "trusted" in body:
        # Trust comes only from the ingest manifest; a probe cannot claim it.
        raise ProbeError("trusted is not a probe field: trust comes only from the manifest")
    cls = body.get("class")
    if type(cls) is not int or cls not in _KEYS:
        raise ProbeError("class must be the integer 1, 2 or 3")
    fields = _KEYS[cls]
    expected = {"class", "store", *fields}
    if set(body) != expected:
        raise ProbeError(f"keys must be exactly {sorted(expected)}")
    if body["store"] != _STORE[cls]:
        raise ProbeError(f"class {cls} goes to store {_STORE[cls]}")
    for k in fields:
        v = body[k]
        if not isinstance(v, str) or not v or len(v) > MAX_TEXT:
            raise ProbeError(f"{k} must be a non-empty string of at most {MAX_TEXT} characters")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    pid = "probe-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {"id": pid, **body}


class ProbeLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def counted(self, session: int) -> int:
        # Read from the file each time, so a restarted server resumes the count.
        if not self.path.exists():
            return 0
        rows = (json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line)
        return sum(1 for r in rows if r["session"] == session and r["counted"])


def create_app(gate, counter: CountingChat, log: ProbeLog, session: int, cap: int) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def left():
        return cap - log.counted(session)

    @app.get("/status")
    def status():
        return {"session": session, "probes_left": left()}

    @app.post("/probe")
    async def probe(request: Request):
        try:
            entry = parse_probe(await request.json())
        except ProbeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except ValueError:
            return JSONResponse({"error": "body is not JSON"}, status_code=400)

        item, store = to_gate_item(entry)
        counter.reset()
        counter.refuse = log.counted(session) >= cap
        try:
            verdict = gate.screen_batch([item], store).verdicts[0]
        except CapReached:
            verdict = None
        row = {"ts": datetime.now(timezone.utc).isoformat(), "session": session,
               "class": entry["class"], "store": store,
               "item_hash": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
               "fields": {k: entry[k] for k in _KEYS[entry["class"]]},
               "cached": False, "counted": False, "admitted": None, "action": None}

        if counter.refused:
            log.append({**row, "status": 429, "calls": 0})
            return JSONResponse({"error": "probe cap reached"}, status_code=429)
        if "reduced assurance" in verdict.reason:
            # Same test as eval/attack_screen.py::row_from. A degraded verdict is
            # not the frozen gate's answer, so it is neither returned nor counted.
            log.append({**row, "status": 503, "calls": counter.calls})
            return JSONResponse({"error": "gate degraded, retry later"}, status_code=503)

        cached = counter.calls == 0
        admitted = verdict.action is Action.ADMIT
        log.append({**row, "status": 200, "calls": counter.calls, "cached": cached,
                    "counted": not cached, "admitted": admitted, "action": verdict.action.name})
        return {"admitted": admitted, "probes_left": left(), "session": session}

    return app


def main() -> None:
    session = os.getenv("PROBE_SESSION")
    if session not in ("1", "2", "3"):
        raise SystemExit("PROBE_SESSION must be 1, 2 or 3")
    try:
        require_frozen()
    except FreezeError as e:
        raise SystemExit(str(e))
    reference = default_reference(real_data_dir())
    if len(reference) == 0:
        raise SystemExit("trusted reference is empty: re-ingest with trust flags first (Task 43)")
    counter = CountingChat(llm.chat)
    gate = default_read_gate(Path(PROBE_DIR), chat=counter, reference=reference)
    log = ProbeLog(Path(PROBE_DIR) / "probes.jsonl")
    app = create_app(gate, counter, log, int(session), PROBE_CAP)
    uvicorn.run(app, host="127.0.0.1", port=PROBE_PORT)


if __name__ == "__main__":
    main()
