# Trust Gate

Capstone fork of `self-correcting-rag`. Goal: screen RAG documents and agent
memory records for poisoned content — write-time and read-time — before an
LLM trusts and uses them. See `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md`.

## Host: Self-Correcting RAG

Agentic RAG over a corpus of AI-security papers, with a Corrective-RAG (CRAG)
self-correcting retrieval loop, hybrid retrieval + reranking, grounded
citations, a refusal guardrail, and a RAGAS evaluation scorecard.

## Architecture

```
question → decompose (LLM: 1–3 sub-questions)
  → per sub-question, CRAG graph:
      retrieve (dense + BM25 → RRF → cross-encoder rerank)
      → grade (LLM: sufficient / weak)
      → [weak, retries<1] rewrite → retrieve (once)
      → [still weak] web_fallback (DuckDuckGo) → re-grade
      → generate (cited answer) | refuse ("Not grounded in the corpus.")
  → synthesize (LLM: merge sub-answers + citations)
→ answer + citations + path-trace
```

- **LLM:** Gemini `gemini-flash-lite-latest` via the OpenAI-compatible Google
  AI Studio endpoint (free tier). The client is **provider-neutral** — set
  `LLM_BASE_URL` / `LLM_MODEL` + the key env to point at any OpenAI-compatible
  provider (Groq, OpenAI, local) without touching call sites.
- **Embeddings/rerank:** local `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`.
- **Vector store:** Chroma (local); sparse: `rank_bm25`.
- **Orchestration:** LangGraph. **UI:** Gradio. **Eval:** RAGAS.

The CRAG loop is built as discrete LangGraph nodes over an explicit `RAGState`,
so new nodes (web-search fallback, refusal) are drop-in additions rather than
rewrites; decomposition and synthesis wrap the graph in `app/orchestrator.py`.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env             # add your GEMINI_API_KEY (https://aistudio.google.com/apikey)
python -m ingest.run_ingest      # one-time: download + chunk + embed + index the corpus
python app.py                    # launch the Gradio UI
python -m eval.run_eval          # produce the eval scorecard (EVAL_N=10 for a quick pass)
pytest                           # run the test suite
```

## Evaluation

RAGAS metrics (Gemini as judge, local MiniLM embeddings) over the Q&A set in
`eval/qa_set.json`. Current scorecard (`eval/results-n10.md`), measured on the
first 10 questions against the full decomposition + web-fallback pipeline:

| Metric | Score |
|---|---|
| faithfulness | 0.97 |
| answer_relevancy | 0.76 |
| context_precision | 0.75 |

For reference, the earlier single-loop CRAG pipeline scored 0.97 / 0.83 / 0.77
over all 20 questions (`eval/results.md`). Faithfulness is unchanged; the dips
in relevancy and precision are expected, since a synthesized multi-sub-answer
and occasional web snippets give the judge more surface to mark down than a
single-loop corpus answer did.

> **Note:** the two scorecards are not directly comparable — different pipelines
> *and* different sample sizes (10 vs 20). The agentic run is capped at 10
> because a full 20-question pass exceeds the Gemini free tier's daily request
> quota: the pipeline issues ~10 generation calls per question (decompose + up
> to 3 sub-queries × grade/rewrite/re-grade/generate + synthesize) on top of the
> judge's own calls.

Reproduce with `EVAL_N=10 python -m eval.run_eval` (`EVAL_N` caps the run size;
omit it for the full set if you have quota headroom). Set `LLM_MIN_INTERVAL` to
pace requests under the per-minute cap — eval defaults it to 4s.

## Limitations

- CRAG loop does one rewrite+retry per sub-question.
- Sub-queries run sequentially (parallel fan-out is future work).
- Decomposition capped at 3 sub-questions.
- Web fallback uses keyless DuckDuckGo (occasional rate-limit/HTML drift); results are re-graded and refused if weak.
- Web answers are labeled `[web: domain]` and are not peer-reviewed sources.
- Corpus is ~22 papers; out-of-scope questions are refused, not answered.
  Questions whose keywords collide with *example* text inside a paper (e.g. a
  benign recipe used as a jailbreak demo) can retrieve a real matching chunk and
  be answered from it — grounded, but not the paper's actual subject.
- Free-tier request limits pace the eval run; it is serialized to stay within
  budget.

## Roadmap

Multi-turn memory, tracing panel, larger corpus + eval set.
