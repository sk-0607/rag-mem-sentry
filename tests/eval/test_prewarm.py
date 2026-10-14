import json
import re

from app.graph import doc_to_item
from eval.prewarm import chunks_for, prewarm
from gate.read import default_read_gate
from gate.reference import ReferenceIndex

TRUSTED = {"id": "ref::0", "paper_id": "ref", "text": "A trusted reference passage.", "trusted": True}


class FakeRetriever:
    def __init__(self, by_query):
        self.by_query = by_query

    def retrieve(self, query, k=5):
        return [dict(d) for d in self.by_query.get(query, [])][:k]


class CountingChat:
    """Answers the judge (every item clean) and S3 extraction (no claims)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, model=None, **kw):
        self.calls += 1
        ids = re.findall(r'<item id="(i\d+)"', messages[1]["content"])
        if "extract" in messages[0]["content"]:
            return json.dumps([{"id": i, "claims": []} for i in ids])
        return json.dumps([{"id": i, "suspicion": 0.0, "label": "clean", "evidence_span": None} for i in ids])


def doc(n, trusted=False):
    return {"id": f"d{n}", "text": f"Untrusted passage number {n}.", "trusted": trusted}


def test_chunks_are_deduped_across_questions_in_first_seen_order():
    r = FakeRetriever({"q1": [doc(0), doc(1)], "q2": [doc(1), doc(2)]})
    assert [c["id"] for c in chunks_for(["q1", "q2"], r, k=10)] == ["d0", "d1", "d2"]


def test_k_caps_chunks_per_question():
    r = FakeRetriever({"q": [doc(i) for i in range(8)]})
    assert len(chunks_for(["q"], r, k=3)) == 3


def test_doc_to_item_trusts_only_boolean_true():
    assert doc_to_item({"id": "a", "text": "x", "trusted": True}).metadata == {"trusted": True}
    assert doc_to_item({"id": "a", "text": "x", "trusted": "true"}).metadata == {"trusted": False}
    assert doc_to_item({"id": "a", "text": "x"}).metadata == {"trusted": False}


def test_warm_cache_makes_zero_gate_llm_calls(tmp_path):
    docs = [doc(i) for i in range(10)]
    retriever = FakeRetriever({"demo question": docs})
    chat = CountingChat()
    gate = default_read_gate(tmp_path, chat=chat, reference=ReferenceIndex([TRUSTED]))

    assert prewarm(["demo question"], retriever, gate, k=10) == 10
    assert chat.calls > 0

    # A live query retrieves the top 5 of what was pre-warmed: all cache hits.
    before = chat.calls
    live = default_read_gate(tmp_path, chat=chat, reference=ReferenceIndex([TRUSTED]))
    live.screen_batch([doc_to_item(d) for d in retriever.retrieve("demo question", k=5)], "corpus")
    assert chat.calls == before
