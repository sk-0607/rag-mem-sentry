import pytest

from eval.ablation import SUBSETS, ablation_table, recompute_action
from gate.policy import DEFAULT_POLICY
from gate.score import aggregate
from gate.types import SignalResult


def row(cls, scores, store="corpus", reduced=False, reasons=None):
    reasons = reasons or {}
    return {"id": "x", "class": cls, "group": "g", "store": store, "action": "admit",
            "trust": 1.0, "top_signal": None, "reduced_assurance": reduced,
            "signals": [{"signal": n, "score": s, "reason": reasons.get(n, "r")}
                        for n, s in scores.items()]}


@pytest.mark.parametrize("scores", [
    {"structural": 0.0, "judge": 0.0, "corroborate": 0.0},
    {"structural": 0.45, "judge": 0.3, "corroborate": 0.0},
    {"structural": 0.0, "judge": 0.6, "corroborate": 0.9},
    {"structural": 0.0, "judge": 0.95, "corroborate": 0.0},
    {"structural": 0.8, "judge": 0.9, "corroborate": 0.9},
])
@pytest.mark.parametrize("store", ["corpus", "experience"])
def test_recompute_full_set_reproduces_the_stored_action(scores, store):
    expected = DEFAULT_POLICY.decide(
        aggregate([SignalResult(n, s, "r") for n, s in scores.items()]), store).action.value
    got = recompute_action(row(1, scores)["signals"], SUBSETS[-1], store)
    assert got == expected


def test_single_signal_subset_ignores_the_other_signals():
    sigs = row(1, {"structural": 0.0, "judge": 0.95})["signals"]
    assert recompute_action(sigs, ("structural",), "corpus") == "admit"
    assert recompute_action(sigs, ("judge",), "corpus") == "block"


def test_absent_signal_counts_as_abstention():
    sigs = row(1, {"structural": 0.9})["signals"]
    assert recompute_action(sigs, ("corroborate",), "corpus") == "admit"


def test_counts_per_class_and_subset():
    rows = [row(1, {"structural": 0.0, "judge": 0.95}),
            row(1, {"structural": 0.0, "judge": 0.0}),
            row(2, {"structural": 0.0, "judge": 0.0, "corroborate": 0.9})]
    t = ablation_table(rows)["table"]
    assert t["judge"]["1"] == {"caught": 1, "scored": 2}
    assert t["structural"]["1"] == {"caught": 0, "scored": 2}
    assert t["corroborate"]["2"] == {"caught": 1, "scored": 1}
    assert t["structural+judge+corroborate"]["3"] == {"caught": 0, "scored": 0}


def test_subset_labels_and_order():
    assert list(ablation_table([])["table"]) == [
        "structural", "judge", "corroborate", "structural+judge",
        "structural+judge+corroborate"]


def test_reduced_assurance_rows_excluded_and_counted():
    out = ablation_table([row(1, {"judge": 0.95}, reduced=True), row(1, {"judge": 0.95})])
    assert out["excluded_reduced_assurance"] == 1
    assert out["table"]["judge"]["1"] == {"caught": 1, "scored": 1}


def test_clean_rows_not_in_class_columns():
    out = ablation_table([row("clean", {"judge": 0.95}, store="experience")])
    assert all(c["scored"] == 0 for cells in out["table"].values() for c in cells.values())
    assert out["excluded_reduced_assurance"] == 0


def test_s3_not_checked_breakdown():
    rows = [row(2, {"structural": 0.0}),
            row(2, {"corroborate": 0.0}, reasons={"corroborate": "no_claims"}),
            row(2, {"corroborate": 0.0}, reasons={"corroborate": "not_covered"}),
            row(2, {"corroborate": 0.0}, reasons={"corroborate": "supported"})]
    assert ablation_table(rows)["s3_not_checked"]["2"] == {"absent": 1, "no_claims": 1, "not_covered": 1}


def test_experience_rows_recomputed_with_the_experience_band():
    sigs = row(3, {"judge": 0.35}, store="experience")["signals"]
    assert recompute_action(sigs, ("judge",), "experience") == "quarantine"
    assert recompute_action(sigs, ("judge",), "corpus") == "admit"
