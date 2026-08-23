"""Experience store: a Chroma similarity index plus a JSONL mirror.

Embedding is on `task` text only — retrieval asks "what past task resembles
this one?", so approach and outcome are payload, not key. The JSONL mirror
exists for the same reason data/chunks.jsonl does: inspectability without
opening Chroma, and it is the source of truth for enumeration.
"""
import json
from pathlib import Path

import chromadb

from app import config
from gate.screen import WriteGate
from gate.types import Action, Verdict
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

    def write(self, record: ExperienceRecord, gate: WriteGate | None = None) -> Verdict | None:
        if gate is not None:
            verdict = gate.screen(record.to_item(), "experience")
            if verdict.action is not Action.ADMIT:
                return verdict
            self._persist(record)
            return verdict
        self._persist(record)
        return None

    def _persist(self, record: ExperienceRecord) -> None:
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

    def retrieve(self, task: str, k: int = config.EXPERIENCE_K) -> list[ExperienceRecord]:
        # Guard the empty case ourselves: Chroma's behavior on an empty
        # collection is version-dependent and not worth relying on.
        if self.collection.count() == 0:
            return []
        embeddings = self.embedder.encode([task])
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        result = self.collection.query(query_embeddings=embeddings, n_results=k)
        ids = result["ids"][0]
        # Chroma stores only `task`; rehydrate the full payload from the
        # JSONL mirror, preserving Chroma's order (the similarity ranking).
        # Skip, don't raise, on an id Chroma knows and the mirror doesn't: the
        # two can drift (a trimmed mirror, a repointed EXPERIENCES_PATH), and
        # read-time posture is degrade rather than refuse — losing one memory
        # must not take the whole answer down with it.
        by_id = {record.id: record for record in self.all_records()}
        return [by_id[rid] for rid in ids if rid in by_id]

    def purge(self, record_ids) -> int:
        """Delete records by id from both Chroma and the JSONL mirror.

        Genuinely destructive, unlike gate/quarantine.py's reversible hold —
        the use it exists for is removing deliberately poisoned demo records
        before a live demo, where leaving them recoverable is the risk. Returns
        how many were actually present, so a caller can tell a real purge from
        a typo'd id.
        """
        targets = set(record_ids)
        existing = self.all_records()
        kept = [r for r in existing if r.id not in targets]
        removed = len(existing) - len(kept)
        if removed == 0:
            return 0
        self.collection.delete(ids=list(targets))
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for record in kept:
                f.write(json.dumps(record.__dict__) + "\n")
        return removed

    def all_records(self) -> list[ExperienceRecord]:
        if not self.jsonl_path.exists():
            return []
        with open(self.jsonl_path, encoding="utf-8") as f:
            return [ExperienceRecord(**json.loads(line)) for line in f if line.strip()]


def default_experience_store(data_dir) -> ExperienceStore:
    """Assemble the standard experience store under `data_dir`.

    The counterpart to memory.gate.default_experience_gate, and the one reader
    of EXPERIENCE_COLLECTION / EXPERIENCES_PATH — before this, both constants
    were declared and never read, so the store's real wiring lived only in
    whatever ad hoc script last built one.

    Paths are wrapped in Path() rather than passed through: _append_jsonl calls
    .parent on jsonl_path, so a bare str from config would not fail until the
    first write, far from the construction that caused it.
    """
    data_dir = Path(data_dir)
    return ExperienceStore(
        chroma_dir=data_dir / Path(config.CHROMA_DIR).name,
        jsonl_path=data_dir / Path(config.EXPERIENCES_PATH).name,
        collection=config.EXPERIENCE_COLLECTION,
    )
