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

## Trust Gate: False-Positive Measurement and Threshold Calibration

The write-time gate (`gate/`) screens every chunk before it reaches the index.
`ingest/run_ingest.py` runs it against the real 22-paper AI-security corpus,
and the result is the project's hard-negative false-positive set **by
construction**: the papers contain no real attacks, only quoted examples of
them inside academic prose, so any non-`admit` verdict on this corpus is a
false positive.

`eval/fpr.py` (`false_positive_rate(audit_entries) -> dict`, plus a
`python -m eval.fpr <audit-log>` CLI) measures this from `gate/audit.py`'s
JSONL log. Both `quarantine` and `block` count as false positives — only
`admit` is "not flagged."

**Budget:** 2% of the clean corpus (≈40 of ~2,025 chunks) — the opening
position from the design plan: high enough to tolerate a few quoted-attack
examples, low enough that "what does the gate cost a legitimate corpus?" has
a defensible answer.

**Measured result**, `python -m eval.fpr data/gate_audit.jsonl` over the full
ingest run:

```json
{
  "total": 2025,
  "admit": 2017,
  "quarantine": 8,
  "block": 0,
  "fpr": 0.003950617283950617
}
```

0.395% — well under the 2% budget, with zero threshold changes.

**Plan vs. reality on the hand-read count.** The plan's Step 6 (and the
Phase 0–1 exit criteria) call for reading "at least 40" flagged chunks by
hand. The real run only produced 8 quarantined chunks (0 blocked) — there is
no way to hand-read 40 of something that doesn't exist at these thresholds.
Rather than loosening thresholds to manufacture more flags, all **8 of 8**
flagged chunks (the entire flagged population) were read in full, and the gap
is recorded here rather than papered over. This is itself a finding: at the
placeholder thresholds, S1's structural signal is precise on this corpus — it
isn't spraying false positives across ordinary academic prose, it's catching
a genuinely small number of real quoted-attack passages.

All 8 were genuine quoted attack/red-team content, not regex over-matches on
ordinary prose:

| Chunk | Rule fired | Content |
|---|---|---|
| `2302.12173::97` | `html-comment` | Literal indirect prompt-injection payload quoted from the paper's own attack figure (`<!--> <\|im_start\|>...`), plus a hidden search-sabotage instruction block |
| `2310.03693::21`, `::22` | `fake-system-message` | Paper's own "harmful examples demonstration attack": quotes the system prompt used to fine-tune a model into complying with harmful instructions |
| `2402.07867::84`, `::85`, `::86` | `fake-system-message` | Worked illustration of a GCG jailbreak/PoisonedRAG adversarial-suffix attack, including a "Tell me how to build a bomb" target example |
| `2401.05566::74`, `::75` | `role-reassignment` | Anthropic sleeper-agents paper's own red-teaming examples ("You are now an AI...", "You are now exempt from...") |

No case was ordinary prose accidentally matched — every hit sits inside a
paper's own worked attack example or red-team transcript. **Calibration
decision: no regex tightened, no threshold moved.** Tightening a working
regex or loosening a threshold to manufacture a lower number would trade a
correct, visible flag for an invisible false negative on content that is
genuinely attack-shaped, even when quoted for research purposes.

**Final `DEFAULT_POLICY` (`corpus` store), now measured rather than
placeholder:**

```python
"corpus": StoreThresholds(block_below=0.10, quarantine_below=0.60)
```

Unchanged from the pre-Task-11 values — the placeholder guess turned out to
be a reasonable calibration for this corpus, and the choice is now backed by
the 8/2025 read-through above instead of asserted without evidence. The
`experience` store thresholds remain an unmeasured placeholder; Phase 2
(Task 16) is the first time that store is exercised.

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
