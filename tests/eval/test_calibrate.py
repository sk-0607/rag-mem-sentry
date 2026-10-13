import json

import pytest

from eval import calibrate
from eval.calibrate import (
    PRIMARY_CLASSES, band_report, load_calibration_rows, load_r5_rows,
    propose_hard_threshold, signal_scores,
)
from gate.policy import DEFAULT_POLICY


def row(cls="clean", store="corpus", action="admit", trust=1.0, signals=(), reduced=False, id="r"):
    return {"id": id, "class": cls, "group": "g", "store": store, "action": action,
            "trust": trust, "top_signal": None, "reduced_assurance": reduced,
            "signals": [{"signal": n, "score": s, "reason": ""} for n, s in signals]}


def test_threshold_is_lowest_attack_score_above_every_clean_score():
    assert propose_hard_threshold([0.3, 0.7, 0.9], [0.0, 0.2, 0.5]) == 0.7


def test_threshold_none_when_not_separable():
    assert propose_hard_threshold([0.5, 0.9], [0.1, 0.9]) is None


def test_threshold_none_without_clean_scores():
    assert propose_hard_threshold([0.7], []) is None


def test_threshold_none_without_attack_scores():
    assert propose_hard_threshold([], [0.2]) is None


def test_zero_attack_score_never_proposed():
    assert propose_hard_threshold([0.0, 0.0], []) is None
    assert propose_hard_threshold([0.0, 0.0], [0.0]) is None


def test_signal_scores_skip_rows_where_the_signal_abstained():
    rows = [row(signals=[("judge", 0.4)]), row(signals=[("structural", 0.9)])]
    assert signal_scores(rows, "judge") == [0.4]


def test_primary_classes():
    assert PRIMARY_CLASSES == {"judge": (1, 3), "corroborate": (2,)}


def test_attack_scores_use_primary_classes_only(tmp_path):
    cal = {"split": "calibration", "model": "m", "policy_commit": "c", "rows": [
        row(cls=1, signals=[("corroborate", 0.95), ("judge", 0.8)]),
        row(cls=2, signals=[("corroborate", 0.7)]),
    ]}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    out = calibrate.main(calibration=p, clean_experience=_empty(tmp_path, "e"),
                         s3_fpr=_empty(tmp_path, "s", split=None,
                                       rows=[row(signals=[("corroborate", 0.1), ("judge", 0.1)])]),
                         r5=tmp_path / "none.jsonl")
    assert out["corroborate"]["proposed"] == 0.7
    assert out["corroborate"]["attack_n"] == 1
    assert out["judge"]["proposed"] == 0.8


def _empty(tmp_path, name, split="clean_experience", rows=()):
    p = tmp_path / f"{name}.json"
    body = {"rows": list(rows)}
    if split:
        body["split"] = split
    p.write_text(json.dumps(body))
    return p


def test_band_report_within_budget_keeps_current_band():
    clean = [row(trust=1.0, id=str(i)) for i in range(100)]
    rep = band_report(clean, [], "corpus")
    q = DEFAULT_POLICY.thresholds["corpus"].quarantine_below
    assert rep["quarantine_below"] == q
    assert rep["clean_total"] == 100 and rep["clean_non_admit"] == 0
    assert rep["within_budget"] is True
    assert rep["largest_band_within_budget"] == q


def test_band_report_over_budget_proposes_largest_passing_band():
    trusts = [0.1, 0.2, 0.3, 0.4, 0.5]
    clean = [row(action="quarantine", trust=t, id=f"q{i}") for i, t in enumerate(trusts)]
    clean += [row(trust=1.0, id=f"a{i}") for i in range(95)]
    rep = band_report(clean, [], "corpus")
    assert rep["clean_rate"] == 0.05 and rep["within_budget"] is False
    # 2% of 100 = 2 rows may sit below the band: trust < 0.3 holds exactly 2.
    assert rep["largest_band_within_budget"] == 0.3


def test_band_report_without_clean_rows_is_not_a_pass():
    rep = band_report([], [row(cls=1, action="block")], "corpus")
    assert rep["clean_rate"] is None and rep["within_budget"] is None
    assert rep["attack_total"] == 1 and rep["attack_caught"] == 1


def test_band_report_filters_by_store_and_drops_reduced_assurance():
    clean = [row(store="corpus"), row(store="experience", action="quarantine"),
             row(store="corpus", action="quarantine", reduced=True)]
    rep = band_report(clean, [], "corpus")
    assert rep["clean_total"] == 1 and rep["clean_non_admit"] == 0


def test_refuses_a_results_file_from_the_test_split(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"split": "test", "rows": []}))
    with pytest.raises(ValueError):
        load_calibration_rows(p)


def test_missing_r5_audit_returns_empty(tmp_path):
    assert load_r5_rows(tmp_path / "none.jsonl") == []


def test_r5_rows_keep_read_stage_only(tmp_path):
    p = tmp_path / "a.jsonl"
    entries = [
        {"item_id": "x", "store": "corpus", "stage": "read", "action": "admit", "trust": 0.9,
         "reason": "ok", "signals": [{"signal": "judge", "score": 0.1, "reason": "quotes"}]},
        {"item_id": "y", "store": "corpus", "stage": "write", "action": "admit", "trust": 1.0,
         "reason": "ok", "signals": []},
    ]
    p.write_text("\n".join(json.dumps(e) for e in entries))
    rows = load_r5_rows(p)
    assert [r["id"] for r in rows] == ["x"]
    assert rows[0]["class"] == "clean" and rows[0]["store"] == "corpus"
    assert rows[0]["top_signal"] == "judge" and rows[0]["reduced_assurance"] is False


def test_cli_writes_no_file(tmp_path, capsys):
    policy = open("gate/policy.py", "rb").read()
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"split": "calibration", "rows": []}))
    inputs = {"calibration": cal, "clean_experience": _empty(tmp_path, "e"),
              "s3_fpr": _empty(tmp_path, "s", split=None), "r5": tmp_path / "none.jsonl"}
    before = set(tmp_path.iterdir())
    calibrate.main(**inputs)
    assert set(tmp_path.iterdir()) == before
    assert open("gate/policy.py", "rb").read() == policy
    assert json.loads(capsys.readouterr().out)["judge"]["proposed"] is None
