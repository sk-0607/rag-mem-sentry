import json
import chromadb
from sentence_transformers import SentenceTransformer
from app import config
import ingest.run_ingest as ing

def test_index_and_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "CHUNKS_PATH", str(tmp_path / "chunks.jsonl"))

    chunks = [
        {"id": "p::0", "text": "adversarial examples fool classifiers",
         "paper_id": "p", "title": "T", "chunk_index": 0, "trusted": False},
        {"id": "p::1", "text": "retrieval augmented generation grounds answers",
         "paper_id": "p", "title": "T", "chunk_index": 1, "trusted": False},
    ]
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    added = ing.index_chunks(chunks, embedder, client)
    assert added == 2

    col = client.get_collection(config.COLLECTION)
    assert col.count() == 2

    ing.write_chunks_jsonl(chunks, config.CHUNKS_PATH)
    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "p::0"


import pytest


@pytest.mark.parametrize("entry,expected", [
    ({"trusted": True}, True),
    ({}, False),
    ({"trusted": False}, False),
    ({"trusted": "true"}, False),
    ({"trusted": 1}, False),
    ({"trusted": None}, False),
])
def test_source_trusted_only_for_boolean_true(entry, expected):
    assert ing.source_trusted(entry) is expected
