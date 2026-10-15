import pytest
from eval.fpr import dedupe_by_item_id, false_positive_rate


def entries(*actions):
    return [{"action": a, "store": "corpus"} for a in actions]


def test_all_admitted_is_zero_fpr():
    assert false_positive_rate(entries("admit", "admit"))["fpr"] == 0.0


def test_quarantine_counts_as_a_false_positive():
    assert false_positive_rate(entries("admit", "quarantine"))["fpr"] == pytest.approx(0.5)


def test_block_counts_as_a_false_positive():
    assert false_positive_rate(entries("admit", "block"))["fpr"] == pytest.approx(0.5)


def test_counts_are_broken_out_by_action():
    stats = false_positive_rate(entries("admit", "admit", "quarantine", "block"))
    assert stats["total"] == 4
    assert stats["admit"] == 2
    assert stats["quarantine"] == 1
    assert stats["block"] == 1


def test_empty_log_is_zero_not_a_division_error():
    assert false_positive_rate([])["fpr"] == 0.0


def test_dedupe_keeps_only_the_last_entry_per_item_id():
    # Simulates a re-ingest: chunk "a" was admitted on the first run, then
    # quarantined on a second run. Only the second (later) verdict counts.
    log = [
        {"item_id": "a", "action": "admit", "store": "corpus"},
        {"item_id": "b", "action": "admit", "store": "corpus"},
        {"item_id": "a", "action": "quarantine", "store": "corpus"},
    ]
    deduped = dedupe_by_item_id(log)
    assert len(deduped) == 2
    stats = false_positive_rate(deduped)
    assert stats["total"] == 2
    assert stats["quarantine"] == 1
    assert stats["admit"] == 1


def test_experience_verdicts_are_excluded():
    # Both stores share one audit log, so a corpus FPR that counts experience
    # records measures something other than what it claims.
    log = entries("admit", "quarantine") + [
        {"action": "admit", "store": "experience"},
        {"action": "admit", "store": "experience"},
    ]
    stats = false_positive_rate(log)
    assert stats["total"] == 2
    assert stats["admit"] == 1


def test_entry_without_a_store_is_excluded():
    assert false_positive_rate([{"action": "admit"}])["total"] == 0


def test_read_stage_entries_excluded_by_default():
    log = entries("admit") + [{"action": "quarantine", "store": "corpus", "stage": "read"}]
    stats = false_positive_rate(log)
    assert stats["total"] == 1
    assert stats["admit"] == 1


def test_stage_read_counts_only_read_entries():
    log = entries("admit") + [{"action": "quarantine", "store": "corpus", "stage": "read"}]
    stats = false_positive_rate(log, stage="read")
    assert stats["total"] == 1
    assert stats["quarantine"] == 1


def test_entry_without_stage_counts_as_write():
    # Entries written before this change have no "stage" key at all.
    log = [{"action": "admit", "store": "corpus"}]
    assert false_positive_rate(log)["total"] == 1


def test_dedupe_does_not_let_a_read_verdict_replace_a_write_verdict():
    # One audit log holds both stages: a later read-time verdict for a chunk is
    # a different measurement, not a re-ingest, so the write verdict must stay.
    log = [
        {"item_id": "a", "action": "quarantine", "store": "corpus", "stage": "write"},
        {"item_id": "a", "action": "admit", "store": "corpus", "stage": "read"},
    ]
    deduped = dedupe_by_item_id(log)
    assert false_positive_rate(deduped)["quarantine"] == 1
    assert false_positive_rate(deduped, stage="read")["admit"] == 1
