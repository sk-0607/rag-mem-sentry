from eval.run_eval import build_records, write_results

def test_build_records_shape():
    qa = [{"question": "q1", "ground_truth": "g1"}]
    runner = lambda q: {"answer": "a1", "citations": ["[T, chunk#0]"],
                        "path_trace": ["generate"], "contexts": ["ctx1"]}
    recs = build_records(qa, runner)
    assert recs[0]["question"] == "q1"
    assert recs[0]["answer"] == "a1"
    assert recs[0]["ground_truth"] == "g1"
    assert recs[0]["contexts"] == ["ctx1"]

def test_write_results(tmp_path):
    p = tmp_path / "results.md"
    write_results({"faithfulness": 0.81, "answer_relevancy": 0.77,
                   "context_precision": 0.69}, str(p))
    txt = p.read_text(encoding="utf-8")
    assert "faithfulness" in txt and "0.81" in txt
