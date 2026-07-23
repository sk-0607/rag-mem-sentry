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
