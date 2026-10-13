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


import math

import pandas as pd

from eval.run_eval import (make_runner, nan_counts, results_path, run_record_path,
                           with_traces, write_run_record)


def test_results_path_gate_off_default():
    assert results_path({}) == "eval/results.md"
    assert results_path({"EVAL_READ_GATE": "0"}) == "eval/results.md"


def test_results_path_gate_on_when_flag_set():
    assert results_path({"EVAL_READ_GATE": "1"}) == "eval/results-gate-on.md"


def test_make_runner_binds_no_gate_by_default():
    seen = []

    def fake_run(question, read_gate="unset"):
        seen.append(read_gate)
        return {"answer": "a"}

    def factory():
        raise AssertionError("gate must not be built when the flag is off")

    assert make_runner({}, fake_run, factory)("q") == {"answer": "a"}
    assert seen == [None]


def test_make_runner_binds_gate_when_flag_set():
    seen, built = [], []
    gate = object()

    def fake_run(question, read_gate="unset"):
        seen.append(read_gate)
        return {"answer": "a"}

    def factory():
        built.append(1)
        return gate

    runner = make_runner({"EVAL_READ_GATE": "1"}, fake_run, factory)
    runner("q1")
    runner("q2")
    assert built == [1]  # one gate (one cache) for the whole run, not one per question
    assert seen == [gate, gate]


def test_with_traces_records_each_question_and_passes_result_through():
    runner, traces = with_traces(lambda q: {"answer": "a", "path_trace": [f"sq0:{q}", "sq0:gate:0/5"]})
    assert runner("q1")["answer"] == "a"
    runner("q2")
    assert traces == [{"question": "q1", "path_trace": ["sq0:q1", "sq0:gate:0/5"]},
                      {"question": "q2", "path_trace": ["sq0:q2", "sq0:gate:0/5"]}]


def test_nan_counts_counts_per_metric_and_treats_an_absent_metric_as_all_missing():
    df = pd.DataFrame({"faithfulness": [0.5, math.nan, math.nan],
                       "answer_relevancy": [0.5, 0.6, 0.7]})
    assert nan_counts(df, ["faithfulness", "answer_relevancy", "context_precision"]) == {
        "faithfulness": 2, "answer_relevancy": 0, "context_precision": 3}


def test_run_record_path_follows_the_gate_flag():
    assert run_record_path({}) == "data/phase3_ragas/run-off.json"
    assert run_record_path({"EVAL_READ_GATE": "1"}) == "data/phase3_ragas/run-on.json"


def test_write_run_record_flags_a_nan_run_incomplete(tmp_path):
    import json
    p = tmp_path / "sub" / "run.json"
    traces = [{"question": "q", "path_trace": ["sq0:gate:0/5"]}]
    write_run_record(str(p), traces, {"faithfulness": 0.5}, {"faithfulness": 1})
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["complete"] is False
    assert rec["nan_counts"] == {"faithfulness": 1}
    assert rec["traces"] == traces
    write_run_record(str(p), traces, {"faithfulness": 0.5}, {"faithfulness": 0})
    assert json.loads(p.read_text(encoding="utf-8"))["complete"] is True
