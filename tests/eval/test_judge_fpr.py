from eval.judge_fpr import hard_negative_items, summarize
from gate.types import Item


class RecordingRetriever:
    def __init__(self, by_question):
        self.by_question = by_question
        self.seen = []

    def retrieve(self, query, k=5):
        self.seen.append((query, k))
        return self.by_question[query]


def chunk(cid):
    return {"id": cid, "text": f"text of {cid}"}


def test_retrieval_uses_raw_questions():
    # Decomposition would add an LLM call and change which chunks the gate sees.
    r = RecordingRetriever({"q one": [chunk("a")], "q two": [chunk("b")]})
    hard_negative_items(["q one", "q two"], r, [])
    assert [q for q, _ in r.seen] == ["q one", "q two"]


def test_dedupes_by_chunk_id_across_questions():
    r = RecordingRetriever({"q1": [chunk("a"), chunk("b")], "q2": [chunk("b"), chunk("c")]})
    items = hard_negative_items(["q1", "q2"], r, [])
    assert [i.id for i in items] == ["a", "b", "c"]
    assert all(i.kind == "corpus" for i in items)


def test_held_items_included_once_even_if_retrieved():
    r = RecordingRetriever({"q1": [chunk("a"), chunk("b")]})
    held = [Item(id="b", text="text of b", kind="corpus"), Item(id="h", text="held", kind="corpus")]
    items = hard_negative_items(["q1"], r, held)
    assert [i.id for i in items] == ["a", "b", "h"]


def entry(action, signals=(), stage="read", item_id="x"):
    return {
        "item_id": item_id, "store": "corpus", "stage": stage, "action": action,
        "signals": [{"signal": n, "score": s, "reason": why, "evidence_span": None}
                    for n, s, why in signals],
    }


def test_summarize_counts_only_read_stage():
    log = [entry("admit"), entry("quarantine", stage="write")]
    stats = summarize(log)
    assert stats["total"] == 1
    assert stats["quarantine"] == 0


def test_summarize_by_top_signal():
    log = [
        entry("quarantine", [("structural", 0.9, "pattern"), ("judge", 0.3, "quotes")]),
        entry("quarantine", [("structural", 0.1, "pattern"), ("judge", 0.8, "instructs")]),
        entry("admit", [("judge", 0.1, "clean")]),
    ]
    assert summarize(log)["by_top_signal"] == {"structural": 1, "judge": 1}


def test_summarize_judge_labels():
    log = [
        entry("admit", [("judge", 0.1, "clean")]),
        entry("admit", [("judge", 0.2, "quotes")]),
        entry("quarantine", [("structural", 0.9, "pattern"), ("judge", 0.8, "quotes")]),
    ]
    assert summarize(log)["judge_labels"] == {"clean": 1, "quotes": 2}
