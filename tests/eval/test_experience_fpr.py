import pytest
from eval.experience_fpr import flagged_experience_report


def e(action, store="experience"):
    return {"action": action, "store": store}


def test_corpus_entries_are_excluded():
    r = flagged_experience_report([e("quarantine", store="corpus"), e("admit")])
    assert r["total"] == 1


def test_all_admitted_is_zero_flag_rate():
    assert flagged_experience_report([e("admit"), e("admit")])["flag_rate"] == 0.0


def test_quarantine_counts_as_flagged():
    assert flagged_experience_report([e("admit"), e("quarantine")])["flag_rate"] == pytest.approx(0.5)


def test_block_counts_as_flagged():
    assert flagged_experience_report([e("admit"), e("block")])["flag_rate"] == pytest.approx(0.5)


def test_counts_broken_out_by_action():
    r = flagged_experience_report([e("admit"), e("admit"), e("quarantine"), e("block")])
    assert (r["total"], r["admit"], r["quarantine"], r["block"]) == (4, 2, 1, 1)


def test_no_experience_entries_is_zero_not_a_division_error():
    assert flagged_experience_report([e("admit", store="corpus")])["flag_rate"] == 0.0


def test_read_stage_entries_excluded_by_default():
    r = flagged_experience_report([e("admit"), {"action": "quarantine", "store": "experience", "stage": "read"}])
    assert r["total"] == 1
    assert r["admit"] == 1


def test_stage_read_counts_only_read_entries():
    log = [e("admit"), {"action": "quarantine", "store": "experience", "stage": "read"}]
    r = flagged_experience_report(log, stage="read")
    assert r["total"] == 1
    assert r["quarantine"] == 1


def test_entry_without_stage_counts_as_write():
    # Entries written before this change have no "stage" key at all.
    assert flagged_experience_report([{"action": "admit", "store": "experience"}])["total"] == 1
