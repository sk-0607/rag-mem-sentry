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


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_file_text(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        # A .docx is a zip of XML; the stdlib reads it, so no new dependency.
        # Legacy binary .doc has no such route and is rejected below.
        import zipfile
        from xml.etree import ElementTree
        root = ElementTree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
        return "\n".join("".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p"))
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unsupported file type {suffix or '(none)'!r} for {path}: use .pdf, .docx, .txt or .md")


def source_text(entry):
    if "path" in entry:
        return read_file_text(entry["path"])
    return download_pdf_text(entry["arxiv_id"])


def source_id(entry):
    # A local file is named by its manifest `id`, else its file stem; it becomes
    # the paper_id and the `<id>::<n>` chunk-id prefix, so keep it unique.
    if "path" in entry:
        return entry.get("id") or Path(entry["path"]).stem
    return entry["arxiv_id"]


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
    # A full ingest is a rebuild: upsert alone left a removed source's vectors
    # in the index, crowding the dense candidates. Only this collection goes;
    # experience memory lives in its own. get-then-delete so a first run has
    # nothing to fail on.
    # ponytail: an interrupted run leaves a partial index; re-run ingest to recover.
    client.get_or_create_collection(config.COLLECTION)
    client.delete_collection(config.COLLECTION)

    all_chunks = []
    for p in papers:
        sid = source_id(p)
        try:
            text = source_text(p)
            chunks = chunk_text(text, sid, p["title"])
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
            log.info("indexed %s (%d/%d chunks admitted)", sid, len(admitted_chunks), len(chunks))
        except Exception as e:  # noqa: BLE001 - skip-and-continue is intentional
            log.warning("skipped %s: %s", sid, e)

    write_chunks_jsonl(all_chunks, config.CHUNKS_PATH)
    log.info("total chunks: %d", len(all_chunks))


if __name__ == "__main__":
    main()
