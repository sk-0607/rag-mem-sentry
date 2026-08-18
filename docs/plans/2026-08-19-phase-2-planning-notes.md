# Phase 2 Planning Notes

**Date:** 2026-08-19
**Status:** Carry-forward notes, not a plan. Resolve these when Phase 2's detailed
plan is written (per the roadmap's "each phase gets a detailed plan written at
its start" rule in `docs/plans/2026-08-16-trust-gate-phase-1.md`).

Source: the final whole-branch review of Phase 0-1 (Opus, full diff
`4308617..9fc2eeb`). Findings #1-3 from that review were fixed immediately
(dead duplicate threshold, stale cache key, non-idempotent audit log — see
commits `acaf603`, `a49a2dd`, `31f9de4`). The items below were deferred on
purpose: none block Phase 0-1's exit criteria, but each has a natural landing
point in Phase 2/3 and should not be forgotten.

## 1. Ingest integration has no test coverage

`ingest/run_ingest.py`'s gate wiring — `default_write_gate` construction and
the per-chunk screen-and-filter loop — is only verified by the one manual
10-minute corpus run. `tests/gate/test_screen.py` tests `WriteGate` in
isolation; `tests/test_ingest.py` predates the gate and calls `index_chunks`/
`write_chunks_jsonl` directly, never `main()`.

**Suggested fix, when picked up:** monkeypatch `download_pdf_text` to return a
fixture with one clean and one poisoned passage, point `config.CHUNKS_PATH`/
`CHROMA_DIR` at `tmp_path`, run `main()`, assert the poisoned chunk is absent
from `chunks.jsonl` and present in the quarantine file.

## 2. Empty signal list fails open

`WriteGate(signals=[], ...)` → `aggregate([])` → `trust=1.0` → ADMIT,
silently. Not reachable via `default_write_gate` today (it always registers
`StructuralSignal`), but the gate's default-deny property currently depends
on someone remembering to pass a signal — there's no guard.

**Suggested fix:** raise in `WriteGate.__post_init__` on an empty signal list,
or treat it as fail-closed quarantine. Worth doing before Phase 3 adds a
second signal and the registration path gets less obvious.

## 3. `eval/asr.py` is orphaned in the plan

The Phase 0-1 plan's File Structure table lists `eval/asr.py` — attack-
success-rate harness — and the plan's own Goal statement promises "a measured
attack-success baseline." No task in the 11-task plan ever creates it, and
the Phase 0-1 exit criteria don't mention it either. This is a plan-document
inconsistency, not an implementation gap — Phase 0-1 followed its tasks
correctly.

**Resolve in Phase 2 planning:** either schedule `eval/asr.py` explicitly (it
likely belongs once there's an actual attack corpus to measure against —
Phase 4/5 per the design spec's plan table), or amend the Phase 0-1 plan's
Goal statement to stop promising it for this phase.

## 4. Signal registry never materialized

The Phase 0-1 plan's File Structure table describes `gate/signals/base.py` as
"`Signal` protocol **+ signal registry**," and the design spec (§7.1) promises
"adding a fourth signal is a new file plus a registry entry." Task 3's actual
steps only specified the protocol, `SignalRun`, and `run_signals` — so the
implementation is task-compliant — but no registry exists. `gate/screen.py`'s
`default_write_gate` hardcodes `signals=[StructuralSignal()]`.

**Resolve in Phase 2/3:** build the registry alongside S2 (Task 17-18 in the
spec's roadmap), not after S3 — adding a second signal is the natural forcing
function and the point where hardcoding starts to hurt.

## 5. S1 is narrower than design spec §7.2 describes

Verified by direct probing during the final review: S1 currently misses
homoglyph substitution, base64/encoding smuggling, tool-call/URL bait, and
some hidden-CSS variants (only `display:none`, `font-size:0`,
`visibility:hidden` are covered — e.g. `opacity:0` is not). The spec's §7.2
lists all of these as S1's remit; Tasks 8-9 never asked for them, so this is
a plan-vs-spec gap, not an implementation defect.

**Resolve in Phase 2 planning:** either schedule these as explicit S1
extension tasks, or narrow the spec's §7.2 description to match what's
actually built — otherwise the ablation table in spec §9.2 will attribute S1's
results to a broader signal than the one that exists.

## Also still open (from Phase 0-1 kickoff, unrelated to the final review)

- **R1** — do the teammate's public attack-benchmark artifacts (PoisonedRAG-
  line, memory-poisoning sets) actually download? Not this repo's own
  22-paper corpus (which downloads fine) — a separate, teammate-owned
  dataset. Still unverified.
- **R3** — does the host's existing CRAG `grade` node already catch class-1
  poison? Not part of the Phase 0-1 11-task plan (needs poison samples + LLM
  calls, outside Phase 0-1's zero-LLM constraint). Unmeasured.
