# tests/memory/test_store_retrieve.py
from memory.types import ExperienceRecord
from memory.store import ExperienceStore


class LengthEmbedder:
    """Embeds by text length, so similarity is controllable from the test."""
    def encode(self, texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]


def rec(rid, task):
    return ExperienceRecord(id=rid, task=task, approach="a", outcome="o",
                            success=True, ts="2026-08-19T00:00:00Z")


def store_at(tmp_path):
    return ExperienceStore(chroma_dir=tmp_path / "chroma",
                           jsonl_path=tmp_path / "experiences.jsonl",
                           collection="test_experiences",
                           embedder=LengthEmbedder())


def test_empty_store_returns_empty_list(tmp_path):
    assert store_at(tmp_path).retrieve("anything") == []


def test_retrieve_returns_experience_records(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1", "summarize"))
    out = s.retrieve("summarize")
    assert isinstance(out[0], ExperienceRecord)


def test_retrieve_finds_the_nearest_task(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("short", "ab"))
    s.write(rec("long", "abcdefghijklmnop"))
    assert s.retrieve("abcdefghijklmnop", k=1)[0].id == "long"


def test_retrieve_respects_k(tmp_path):
    s = store_at(tmp_path)
    for i in range(5):
        s.write(rec(f"r{i}", "x" * (i + 1)))
    assert len(s.retrieve("xx", k=2)) == 2


def test_k_larger_than_store_returns_all(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1", "abc"))
    assert len(s.retrieve("abc", k=10)) == 1


def test_retrieved_records_carry_full_payload(tmp_path):
    s = store_at(tmp_path)
    s.write(ExperienceRecord(id="r1", task="t", approach="APPROACHY",
                             outcome="OUTCOMEZ", success=False, ts="2026-08-19T00:00:00Z"))
    got = s.retrieve("t", k=1)[0]
    assert got.approach == "APPROACHY" and got.outcome == "OUTCOMEZ" and got.success is False
