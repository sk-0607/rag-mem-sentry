import json
import os
import pytest
from app.retrieval import rrf_merge

def test_rrf_prefers_items_ranked_high_in_both():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = rrf_merge([dense, sparse])
    assert fused[0] in ("a", "b")
    assert set(fused) == {"a", "b", "c", "d"}

def test_rrf_single_list_preserves_order():
    assert rrf_merge([["x", "y", "z"]]) == ["x", "y", "z"]

@pytest.mark.skipif(not os.path.exists("data/chunks.jsonl"),
                    reason="corpus not built")
def test_retriever_returns_scored_docs():
    from app.retrieval import Retriever
    r = Retriever()
    docs = r.retrieve("what is a prompt injection attack?", k=3)
    assert 1 <= len(docs) <= 3
    assert all("score" in d and d["text"] for d in docs)
    assert all(d["source"] == "corpus" for d in docs)

def _corpus_predates_trust_flags():
    # Ingest stamps `trusted` on every chunk, so one line is representative. A
    # corpus built before that has no field until the owner re-ingests.
    try:
        with open("data/chunks.jsonl", encoding="utf-8") as f:
            return "trusted" not in json.loads(f.readline())
    except (OSError, ValueError):
        return False


@pytest.mark.skipif(not os.path.exists("data/chunks.jsonl"),
                    reason="corpus not built")
@pytest.mark.skipif(_corpus_predates_trust_flags(),
                    reason="built corpus predates trust flags; re-ingest (Task 43) stamps them")
def test_retrieved_chunks_carry_the_stored_trust_flag():
    from app.retrieval import Retriever
    docs = Retriever().retrieve("what is a prompt injection attack?", k=3)
    assert docs and all(isinstance(d.get("trusted"), bool) for d in docs)
