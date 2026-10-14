from ingest.build_nq import assign_trust, group_docs, pick_questions


def test_pick_questions_skips_answer_in_question_and_missing_gold():
    prag = {
        "q1": {"question": "who sang it", "correct answer": "A", "incorrect answer": "B"},
        "q2": {"question": "is it Blue", "correct answer": "Red", "incorrect answer": "blue"},
        "q3": {"question": "no gold here", "correct answer": "C", "incorrect answer": "D"},
        "q4": {"question": "when", "correct answer": "1990", "incorrect answer": "1991"},
    }
    picked = pick_questions(prag, {"q1": ["p1"], "q2": ["p2"], "q4": ["p4"]}, n=5)
    assert [p["qid"] for p in picked] == ["q1", "q4"]
    assert picked[0]["gold"] == ["p1"] and picked[0]["incorrect"] == "B"


def test_group_docs_joins_by_title_and_drops_short_non_gold():
    long = " ".join(["w"] * 25)
    passages = [
        {"_id": "a1", "title": "A", "text": long, "gold": True},
        {"_id": "a2", "title": "A", "text": "short", "gold": False},
        {"_id": "b1", "title": "B", "text": "tiny gold", "gold": True},
        {"_id": "c1", "title": "C", "text": "tiny", "gold": False},
        {"_id": "d1", "title": "D", "text": long + " end", "gold": False},
    ]
    docs = group_docs(passages, min_words=20)
    assert [d["title"] for d in docs] == ["A", "B", "D"]
    assert docs[0]["text"] == long and docs[0]["gold"] is True
    assert docs[1]["text"] == "tiny gold"
    assert docs[2]["gold"] is False


def test_assign_trust_keeps_gold_trusted_and_holds_out_a_fraction():
    docs = [{"title": f"t{i}", "text": "x", "gold": i < 3} for i in range(23)]
    entries = assign_trust(docs, untrusted_frac=0.1, seed=0)
    assert len({e["id"] for e in entries}) == 23
    assert all(e["trusted"] is True for e, d in zip(entries, docs) if d["gold"])
    assert sum(e["trusted"] is False for e in entries) == 2
    assert assign_trust(docs, 0.1, 0) == entries
