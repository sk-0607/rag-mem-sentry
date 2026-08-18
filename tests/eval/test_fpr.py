import pytest
from eval.fpr import dedupe_by_item_id, false_positive_rate


def entries(*actions):
    return [{"action": a} for a in actions]


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
        {"item_id": "a", "action": "admit"},
        {"item_id": "b", "action": "admit"},
        {"item_id": "a", "action": "quarantine"},
    ]
    deduped = dedupe_by_item_id(log)
    assert len(deduped) == 2
    stats = false_positive_rate(deduped)
    assert stats["total"] == 2
    assert stats["quarantine"] == 1
    assert stats["admit"] == 1
