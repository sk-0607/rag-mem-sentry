"""Experience store: a Chroma similarity index plus a JSONL mirror.

Embedding is on `task` text only — retrieval asks "what past task resembles
this one?", so approach and outcome are payload, not key. The JSONL mirror
exists for the same reason data/chunks.jsonl does: inspectability without
opening Chroma, and it is the source of truth for enumeration.
"""
import json

import chromadb

from app import config
from memory.types import ExperienceRecord


class ExperienceStore:
    def __init__(self, chroma_dir, jsonl_path, collection, embedder=None):
        self.chroma_dir = chroma_dir
        self.jsonl_path = jsonl_path
        self.collection_name = collection
        # Lazy: constructed on first use so importing this module, or building
        # a store that only ever calls all_records(), never loads MiniLM.
        self._embedder = embedder
        self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            # get_or_create, not get_collection: unlike the corpus there is no
            # separate ingest step that creates the collection first.
            self._collection = client.get_or_create_collection(self.collection_name)
        return self._collection

    def write(self, record: ExperienceRecord) -> None:
        self._append_jsonl(record)
        embeddings = self.embedder.encode([record.task])
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        self.collection.add(
            ids=[record.id],
            embeddings=embeddings,
            documents=[record.task],
            metadatas=[{"success": record.success, "ts": record.ts}],
        )

    def _append_jsonl(self, record: ExperienceRecord) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def all_records(self) -> list[ExperienceRecord]:
        if not self.jsonl_path.exists():
            return []
        with open(self.jsonl_path, encoding="utf-8") as f:
            return [ExperienceRecord(**json.loads(line)) for line in f if line.strip()]
