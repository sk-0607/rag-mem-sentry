# Self-Correcting RAG

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

RAGAS metrics (Gemini as judge, local MiniLM embeddings) on the fixed
20-question Q&A set in `eval/qa_set.json`. Committed scorecard
(`eval/results.md`):

| Metric | Score |
|---|---|
| faithfulness | 0.97 |
| answer_relevancy | 0.83 |
| context_precision | 0.77 |

> **Note:** these scores were measured on the earlier single-loop CRAG pipeline,
> before query decomposition and the web-search fallback were added. They have
> not yet been re-run against the current orchestrator.

Reproduce with `python -m eval.run_eval` (set `EVAL_N` to cap the run size for a
quick pass on a tight free-tier budget).

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
