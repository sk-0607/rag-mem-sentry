from memory.types import ExperienceRecord
from memory.store import ExperienceStore


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def rec(rid, task="summarize a paper"):
    return ExperienceRecord(id=rid, task=task, approach="a", outcome="o",
                            success=True, ts="2026-08-19T00:00:00Z")


def store_at(tmp_path):
    return ExperienceStore(chroma_dir=tmp_path / "chroma",
                           jsonl_path=tmp_path / "experiences.jsonl",
                           collection="test_experiences",
                           embedder=FakeEmbedder())


def test_purged_record_leaves_all_records(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1"))
    s.write(rec("r2"))
    s.purge(["r1"])
    assert [r.id for r in s.all_records()] == ["r2"]


def test_purged_record_is_no_longer_retrievable(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1", "abc"))
    s.purge(["r1"])
    assert s.retrieve("abc", k=1) == []


def test_purge_returns_the_number_removed(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1"))
    s.write(rec("r2"))
    assert s.purge(["r1", "r2"]) == 2


def test_purging_an_unknown_id_is_a_no_op(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1"))
    assert s.purge(["nope"]) == 0
    assert [r.id for r in s.all_records()] == ["r1"]


def test_purge_survives_reopen(tmp_path):
    store_at(tmp_path).write(rec("r1"))
    store_at(tmp_path).purge(["r1"])
    assert store_at(tmp_path).all_records() == []


def test_purge_on_an_empty_store_does_not_raise(tmp_path):
    assert store_at(tmp_path).purge(["r1"]) == 0
