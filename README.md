# RAG Mem Sentry

Agentic RAG over a BEIR NQ Wikipedia corpus (the AI-security papers the figures before "Final corpus:
BEIR NQ" were measured on were a placeholder; see that section in [docs/results.md](docs/results.md)), with hybrid retrieval +
reranking, cited answers, and a RAGAS evaluation scorecard — plus a
content trust gate that screens documents and agent memory records for
poisoned content, write-time and read-time, before an LLM trusts and uses
them. See `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md`.

## Architecture

```
question → decompose (LLM: 1–3 sub-questions)
  → per sub-question, graph:
      retrieve (dense + BM25 → RRF → cross-encoder rerank)
      → gate (read-time screen; only when a read gate is passed)
      → generate (cited answer)
  → synthesize (LLM: merge sub-answers + citations)
→ answer + citations + path-trace
```

- **LLM:** Gemini `gemini-flash-lite-latest` via the OpenAI-compatible Google
  AI Studio endpoint (free tier) — the default in `app/config.py`, overridable
  per-checkout with `LLM_MODEL` in `.env`. The client is **provider-neutral** —
  set `LLM_BASE_URL` / `LLM_MODEL` + the key env to point at any
  OpenAI-compatible provider (Groq, OpenAI, local) without touching call sites.
  Measured security results name the model they ran against; see [Limitations](docs/limitations.md). Since Phase 4
  the gate's model (judge and S3) is `gemini-flash-lite-latest`, recorded in
  `eval/frozen_gate.json`; the Phase 4–6 trials answered with `gemini-3.1-flash-lite`.
- **Embeddings/rerank:** local `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`.
- **Vector store:** Chroma (local); sparse: `rank_bm25`.
- **Orchestration:** LangGraph. **UI:** Gradio. **Eval:** RAGAS.

The graph is discrete LangGraph nodes over an explicit `RAGState`; decomposition
and synthesis wrap it in `app/orchestrator.py`.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env             # add your GEMINI_API_KEY (https://aistudio.google.com/apikey)
python -m ingest.build_nq <corpus-src>    # one-time: build the NQ corpus and manifest (reads <src>/beir-nq/nq, <src>/PoisonedRAG)
python -m ingest.run_ingest      # one-time: chunk + gate + embed + index the corpus
python app.py                    # launch the Gradio UI
python -m app.web                # or the web UI on http://127.0.0.1:7861 (same flags, same trust panel)
python -m eval.run_eval          # produce the eval scorecard (EVAL_N=10 for a quick pass)
pytest                           # run the test suite
```

**Trust panel.** Two `.env` flags, both off by default, turn on the gated demo in the UI:
`UI_READ_GATE=1` screens every retrieved passage and memory with the read gate before the model
sees it, and `UI_EXPERIENCE_STORE=1` attaches experience memory. Only the exact value `1` turns a
flag on. The read gate uses `data/` (or `GATE_DIR`) and the cache `eval.prewarm` fills there; a
query on an uncached passage spends LLM calls. The panel under the chat shows, for each candidate
the gate screened on that query, its action, trust, top signal with reason, and the text with the
evidence span marked. It reads the gate's audit log, so it is single-user: two queries at once
mix their rows. The web UI (`app/web.py`, `app/static/index.html`) shows the same panel and runs one
query at a time, so its rows do not mix. It binds 127.0.0.1 only and is separate from
`app/api.py`, which does not expose the panel's per-signal scores and reasons.

**Changing the corpus.** `ingest/papers.json` lists the sources. An entry is either an arXiv paper,
`{"arxiv_id": "2005.11401", "title": "...", "trusted": true}`, downloaded from arxiv.org, or a
local file, `{"path": "docs/handbook.pdf", "title": "...", "id": "handbook", "trusted": true}`,
read from disk. Local files may be `.pdf`, `.docx`, `.txt` or `.md`; a legacy binary `.doc` is
skipped with a warning (save it as `.docx`). `id` is optional and defaults to the file name
without its extension; it prefixes every chunk id, so keep it unique. Every ingest is a full
rebuild: it clears the `papers` index first (experience memory, a separate collection, is not
touched), so a source removed from the manifest leaves nothing behind. To swap the trusted
reference set, edit the manifest and re-run ingest; S3's cache misses on the new set by itself.
Ingest refuses a manifest with a duplicate source id, or an entry with neither `arxiv_id` nor
`path`, before it clears the index. Restart a running app after a re-ingest: it holds a handle to
the index that was cleared.

## Documentation

Detailed results and design notes live under `docs/`. Each section heading that used to be in
this README is listed under the file that now holds it, so a pointer to "README section X"
elsewhere in the repository resolves here.

- [Architecture details](docs/architecture.md)
- [Results](docs/results.md)
  - Evaluation
  - Content Gate: False-Positive Measurement and Threshold Calibration
  - R3: Does the Existing CRAG Grader Already Catch Class-1 Poison?
  - Phase 2 Results: Experience-Store Calibration and the Ungated Poisoning Demonstration
    - Experience flag rate
    - Poisoning demonstration — the phase's exit criterion
    - False-fact / unsafe-approach poison retrial (2026-08-24)
    - Gate-off baseline on the simplified host (2026-09-16)
  - Phase 3: R5 — Does the Judge Separate Quoting from Instructing?
    - Hand-read: the 8 non-admits (these are also the 8 held chunks)
    - Hand-read: the 8 other `quotes` labels (admitted)
    - Decision rule
    - What this shows and what it does not
  - Phase 3: Gate-Off Vs Gate-On Harnesses (built; run 2026-10-15 except RAGAS)
  - Phase 4: S3 Corroboration (built and measured)
    - Trust-flagged re-ingest (Task 43, run 2026-10-14)
    - Demo trust marking (2026-10-14)
    - Phase 4: S3 on Clean Untrusted Content (Task 44, run 2026-10-14)
    - Final corpus: BEIR NQ (2026-10-14)
  - Phase 4: Calibration (Task 45, run 2026-10-14)
  - Phase 4: Signal Ablation (Task 46, run 2026-10-14)
  - Phase 4: Class-2 Attack Success (Task 47, run 2026-10-15)
  - Phase 5: Adaptive Round (built and run)
    - Gate freeze (Task 55, run 2026-10-14)
    - Probe history
    - Phase 5: Adaptive Detection (Task 59, run 2026-10-15)
    - Phase 5: Adaptive Class-2 Attack Success (Task 60, run 2026-10-15)
    - Phase 5: What the Adaptive Round Shows
  - Phase 6: Class-1 Attack Success (Task 65, run 2026-10-15)
  - Phase 6: Class-3 Attack Success (Task 66, run 2026-10-15)
  - Phase 6: Overhead (Task 67, run 2026-10-15)
- [Open measurements and known concerns](docs/open-measurements.md)
  - Open measurements
  - Known Phase 3+ concerns (named, not addressed)
- [Limitations](docs/limitations.md)
  - Limitations
- [Roadmap](docs/roadmap.md)
  - Roadmap

Design spec: `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md`. Evaluation report:
`docs/report.md`. Plans: `docs/plans/`.
