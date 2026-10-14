"""Fill the read gate's cache for a demo's questions before the demo.

The gate's verdicts are content-keyed, so a chunk screened here is a cache hit
when a live query retrieves it: the demo then spends no gate LLM call on it.
Spends LLM budget once (S2 one call per 5 uncached chunks, S3 up to 2 per
uncached untrusted chunk); a rerun only pays for what is not cached yet.

Usage: python -m eval.prewarm <questions.txt>   (one question per line)
"""
import os
import sys
from pathlib import Path

from app import config
from app.graph import doc_to_item
from gate.read import default_read_gate

# Wider than TOP_K: the agent retrieves per sub-question, not per question, so
# the live top-5 can differ from the question's own top-5.
# ponytail: covers sub-questions only as far as their chunks rank in the
# question's top PREWARM_K; list the sub-questions in the file if a demo misses.
PREWARM_K = config.FUSE_M


def chunks_for(questions: list[str], retriever, k: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for q in questions:
        for c in retriever.retrieve(q, k=k):
            seen.setdefault(c["id"], c)
    return list(seen.values())


def prewarm(questions: list[str], retriever, gate, k: int = PREWARM_K) -> int:
    chunks = chunks_for(questions, retriever, k)
    gate.screen_batch([doc_to_item(c) for c in chunks], "corpus")
    return len(chunks)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    questions = [q.strip() for q in Path(argv[0]).read_text(encoding="utf-8").splitlines() if q.strip()]
    from app.retrieval import Retriever

    # The demo's read gate must use this same directory, or every lookup misses.
    data_dir = Path(os.getenv("GATE_DIR", "data"))
    n = prewarm(questions, Retriever(), default_read_gate(data_dir))
    print(f"{n} chunks screened for {len(questions)} questions into {data_dir / 'gate_cache.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
