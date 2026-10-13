import json
import random

import pytest

from eval import attack_corpus as ac
from eval.attack_corpus import AttackCorpusError
from tests.eval.corpus_fixture import make_items, write_corpus


def _bad(items):
    with pytest.raises(AttackCorpusError) as e:
        ac.validate(items)
    return str(e.value)


# validator
def test_valid_fixture_passes():
    ac.validate(make_items())


def test_duplicate_id_rejected():
    items = make_items()
    items[1]["id"] = items[0]["id"]
    assert "c1-0" in _bad(items)


def test_unknown_class_rejected():
    for bad in (4, "2"):
        items = make_items()
        items[0]["class"] = bad
        assert "c1-0" in _bad(items)


@pytest.mark.parametrize("idx,field", [(0, "text"), (30, "target_answer"), (60, "approach"),
                                       (0, "check"), (30, "check"), (60, "check")])
def test_missing_field_for_class_rejected(idx, field):
    items = make_items()
    del items[idx][field]
    msg = _bad(items)
    assert items[idx]["id"] in msg and field in msg


def test_empty_string_field_rejected():
    items = make_items()
    items[0]["text"] = ""
    assert "c1-0" in _bad(items)


@pytest.mark.parametrize("val", [True, False])
def test_trusted_key_rejected(val):
    items = make_items()
    items[0]["trusted"] = val
    assert "c1-0" in _bad(items)


def test_group_spanning_two_classes_rejected():
    items = make_items()
    items[0]["group"] = items[30]["group"]
    assert "group" in _bad(items)


def test_fewer_than_thirty_per_class_rejected():
    assert "30" in _bad(make_items(per_class=29))


def test_class2_with_fewer_than_ten_groups_rejected():
    assert "10" in _bad(make_items(class2_groups=9))


def test_target_answer_inside_target_question_rejected():
    items = make_items()
    items[30]["target_question"] = "What is FIXTURE ANSWER 2-0 exactly?"
    assert "c2-0" in _bad(items)


def test_error_lists_every_violation():
    items = make_items()
    del items[0]["text"]
    items[1]["check"] = ""
    msg = _bad(items)
    assert "c1-0" in msg and "c1-1" in msg


# reader
def test_utf8_bom_tolerated(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"id": "a"}\n')
    assert ac.read_items(p) == [{"id": "a"}]


def test_blank_lines_ignored(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('\n{"id": "a"}\n\n   \n{"id": "b"}\n', encoding="utf-8")
    assert len(ac.read_items(p)) == 2


def test_malformed_line_reports_line_number(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a"}\n{nope\n', encoding="utf-8")
    with pytest.raises(AttackCorpusError, match="line 2"):
        ac.read_items(p)


def test_non_object_line_rejected(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text('{"id": "a"}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(AttackCorpusError, match="line 2"):
        ac.read_items(p)


# split
def test_no_group_straddles_splits():
    items = make_items()
    splits = ac.assign_splits(items)
    assert set(splits.values()) <= {"calibration", "test"}
    assert set(splits) == {it["group"] for it in items}


def test_split_is_independent_of_file_order():
    items = make_items()
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    assert ac.assign_splits(items) == ac.assign_splits(shuffled)


def test_each_class_has_both_splits():
    items = make_items()
    splits = ac.assign_splits(items)
    for cls in (1, 2, 3):
        sides = {splits[it["group"]] for it in items if it["class"] == cls}
        assert sides == {"calibration", "test"}


def test_split_alternates_within_class():
    items = make_items(class2_groups=9)
    splits = ac.assign_splits(items)
    sides = [splits[g] for g in {it["group"] for it in items if it["class"] == 2}]
    assert abs(sides.count("calibration") - sides.count("test")) <= 1


# freeze and load
def _frozen(tmp_path, items=None):
    write_corpus(tmp_path, items or make_items())
    return ac.freeze(tmp_path, "2026-01-01")


def test_freeze_writes_manifest(tmp_path):
    items = make_items()
    m = _frozen(tmp_path, items)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == m
    assert set(m) == {"sha256", "frozen", "counts", "groups"}
    assert m["frozen"] == "2026-01-01"
    assert sum(n for c in m["counts"].values() for n in c.values()) == len(items)


def test_freeze_refuses_invalid_corpus_and_writes_nothing(tmp_path):
    write_corpus(tmp_path, make_items(per_class=5))
    with pytest.raises(AttackCorpusError):
        ac.freeze(tmp_path, "2026-01-01")
    assert not (tmp_path / "manifest.json").exists()


def test_load_refuses_unfrozen_corpus(tmp_path):
    write_corpus(tmp_path, make_items())
    with pytest.raises(AttackCorpusError, match="not frozen"):
        ac.load(tmp_path)


def test_load_refuses_changed_file(tmp_path):
    _frozen(tmp_path)
    with open(tmp_path / "items.jsonl", "ab") as f:
        f.write(b"\n")
    with pytest.raises(AttackCorpusError, match="changed since freeze"):
        ac.load(tmp_path)


def test_load_filters_by_split(tmp_path):
    _frozen(tmp_path)
    cal, tst = ac.load(tmp_path, "calibration"), ac.load(tmp_path, "test")
    assert cal and tst and len(cal) + len(tst) == len(ac.load(tmp_path))
    assert {i["split"] for i in cal} == {"calibration"}


def test_load_adds_split_to_every_item(tmp_path):
    _frozen(tmp_path)
    assert all(i["split"] in ("calibration", "test") for i in ac.load(tmp_path))


def test_load_rejects_unknown_split_name(tmp_path):
    _frozen(tmp_path)
    with pytest.raises(AttackCorpusError):
        ac.load(tmp_path, "train")


# projection
def test_corpus_classes_project_to_untrusted_corpus_items():
    items = make_items()
    for e in (items[0], items[30]):
        item, store = ac.to_gate_item(e)
        assert store == "corpus" and item.kind == "corpus"
        assert item.id == e["id"] and item.text == e["text"]
        assert item.metadata == {"trusted": False}


def test_class3_projects_to_experience_item():
    e = make_items()[60]
    item, store = ac.to_gate_item(e)
    assert store == "experience" and item.kind == "experience" and item.id == e["id"]
    assert e["approach"] in item.text


def test_projection_is_deterministic():
    e = make_items()[60]
    a, b = ac.to_gate_item(e)[0], ac.to_gate_item(e)[0]
    assert a.text == b.text and a.metadata == b.metadata
