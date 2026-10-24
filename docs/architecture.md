# Architecture details

The `gate` node is Phase 3's read-time gate (`gate/read.py`). It exists only when a
caller passes `read_gate=` to `run_agentic` / `build_graph`; `None` (the default, and what
`ui.py` and `api.py` pass) builds today's graph with no gate node and no experience filtering.
When on, it screens the retrieved top-5 chunks with S1 plus the S2 policy judge
(`gate/signals/judge.py`), one batched judge call per sub-query, verdicts cached by content
hash, and passes only admitted chunks to `generate`. The same gate filters retrieved experience
records before they are formatted into the prompt. If an LLM signal is unavailable it degrades to
the surviving signals and says so in `path_trace`. A sub-query whose every chunk is excluded reaches `generate`
with an empty context.

Phase 4 adds a third read-time signal, S3 corroboration (`gate/signals/corroborate.py`), so the
read gate now runs S1 + S2 + S3. Trust is set by the owner: each entry in `ingest/papers.json`
may carry `"trusted": true`, and ingest stamps that flag on every chunk of the source. Nothing
else sets it; a `trusted` field inside a document is overwritten, and a chunk with no flag is
read as untrusted. S3 checks an *untrusted* chunk's factual claims against the trusted chunks
only (`gate/reference.py`, a BM25 index over them) and scores by the strongest contradiction. It
never compares a chunk with the other chunks retrieved for the same query, never sees the query,
and abstains on trusted chunks, on the experience store, and when the trusted set is empty. What
it does not check: a claim the trusted set does not cover is reported as not covered, not
guessed at, and the trusted set itself is taken on the owner's word.

> **Pipeline change (2026-09-15):** the inherited CRAG corrective loop (grade →
> rewrite → web fallback) and the refusal guardrail ("Not grounded in the
> corpus.") were removed; every sub-question now goes straight from retrieve to
> generate (or, with a read gate, retrieve → gate → generate). The RAGAS scorecards, R3, and the Phase 2 / poison-trial results in [results.md](results.md)
> were all measured **with** the loop and refusal in place, and describe that
> pipeline, not the current one.
