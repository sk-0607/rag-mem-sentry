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
         "paper_id": "p", "title": "T", "chunk_index": 0},
        {"id": "p::1", "text": "retrieval augmented generation grounds answers",
         "paper_id": "p", "title": "T", "chunk_index": 1},
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
