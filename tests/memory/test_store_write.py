import json
from memory.types import ExperienceRecord
from memory.store import ExperienceStore


class FakeEmbedder:
    """Deterministic 3-dim embedding; no model load."""
    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def rec(rid="r1", task="summarize a paper", approach="retrieve then generate"):
    return ExperienceRecord(id=rid, task=task, approach=approach,
                            outcome="ok", success=True, ts="2026-08-19T00:00:00Z")


def store_at(tmp_path, embedder=None):
    return ExperienceStore(chroma_dir=tmp_path / "chroma",
                           jsonl_path=tmp_path / "experiences.jsonl",
                           collection="test_experiences",
                           embedder=embedder or FakeEmbedder())


def test_write_appends_to_jsonl(tmp_path):
    s = store_at(tmp_path)
    s.write(rec())
    lines = (tmp_path / "experiences.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["id"] == "r1"


def test_jsonl_row_roundtrips_every_field(tmp_path):
    s = store_at(tmp_path)
    s.write(rec())
    row = json.loads((tmp_path / "experiences.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(row) >= {"id", "task", "approach", "outcome", "success", "ts"}


def test_embedding_is_taken_over_task_not_screened_text(tmp_path):
    embedder = FakeEmbedder()
    s = store_at(tmp_path, embedder)
    s.write(rec(task="TASKX", approach="APPROACHY"))
    assert embedder.seen == ["TASKX"]


def test_all_records_returns_written_records(tmp_path):
    s = store_at(tmp_path)
    s.write(rec("r1"))
    s.write(rec("r2"))
    assert [r.id for r in s.all_records()] == ["r1", "r2"]


def test_all_records_survives_reopen(tmp_path):
    store_at(tmp_path).write(rec("r1"))
    assert [r.id for r in store_at(tmp_path).all_records()] == ["r1"]


def test_all_records_on_missing_file_is_empty(tmp_path):
    assert store_at(tmp_path).all_records() == []
