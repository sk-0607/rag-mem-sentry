import io
import json
import logging
import os
from pathlib import Path

import requests
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

from app import config
from app.chunking import chunk_text
from gate.screen import default_write_gate
from gate.types import Action, Item

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ingest")


def download_pdf_text(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=60, headers={"User-Agent": "research-rag/0.1"})
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def source_trusted(entry):
    # `is True`, not truthiness: "false" or 1 in a hand-edited manifest must not
    # silently grant trust.
    return entry.get("trusted") is True


def index_chunks(chunks, embedder, client):
    col = client.get_or_create_collection(config.COLLECTION)
    embeddings = embedder.encode([c["text"] for c in chunks]).tolist()
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"paper_id": c["paper_id"], "title": c["title"],
                    "chunk_index": c["chunk_index"], "trusted": c["trusted"]} for c in chunks],
    )
    return len(chunks)


def write_chunks_jsonl(chunks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def main():
    papers = json.load(open("ingest/papers.json", encoding="utf-8"))
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # Derived from config.CHUNKS_PATH at call time (not import time) so tests
    # can monkeypatch config and redirect the gate's data files along with
    # the chunks/chroma paths, rather than writing into the real data/ dir.
    gate = default_write_gate(Path(config.CHUNKS_PATH).parent)

    all_chunks = []
    for p in papers:
        try:
            text = download_pdf_text(p["arxiv_id"])
            chunks = chunk_text(text, p["arxiv_id"], p["title"])
            # Assigned after chunking and from the manifest only, so nothing a
            # chunk arrives with (or says) can claim trust.
            trusted = source_trusted(p)
            for chunk in chunks:
                chunk["trusted"] = trusted

            admitted_chunks = []
            for chunk in chunks:
                verdict = gate.screen(Item(id=chunk["id"], text=chunk["text"], kind="corpus"), "corpus")
                if verdict.action is not Action.ADMIT:
                    continue  # quarantined or blocked; recorded in the audit log
                admitted_chunks.append(chunk)

            if admitted_chunks:
                index_chunks(admitted_chunks, embedder, client)
            all_chunks.extend(admitted_chunks)
            log.info("indexed %s (%d/%d chunks admitted)", p["arxiv_id"], len(admitted_chunks), len(chunks))
        except Exception as e:  # noqa: BLE001 - skip-and-continue is intentional
            log.warning("skipped %s: %s", p["arxiv_id"], e)

    write_chunks_jsonl(all_chunks, config.CHUNKS_PATH)
    log.info("total chunks: %d", len(all_chunks))


if __name__ == "__main__":
    main()
