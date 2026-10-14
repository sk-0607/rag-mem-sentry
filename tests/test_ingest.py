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


def _docx(path, *paragraphs):
    import zipfile
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_read_file_text_plain(tmp_path, suffix):
    p = tmp_path / f"doc{suffix}"
    p.write_text("# Title\nSome text.", encoding="utf-8")
    assert ing.read_file_text(p) == "# Title\nSome text."


def test_read_file_text_docx_keeps_paragraphs(tmp_path):
    p = tmp_path / "doc.docx"
    _docx(p, "First paragraph.", "Second paragraph.")
    assert ing.read_file_text(p) == "First paragraph.\nSecond paragraph."


@pytest.mark.parametrize("name", ["old.doc", "sheet.xlsx", "noext"])
def test_read_file_text_rejects_unsupported_types(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        ing.read_file_text(p)


def test_read_file_text_strips_a_utf8_bom(tmp_path):
    # A BOM left on the first word trips S1's zero-width rule and quarantines chunk 0.
    p = tmp_path / "doc.txt"
    p.write_bytes("\ufeffHello.".encode("utf-8"))
    assert ing.read_file_text(p) == "Hello."


def test_check_source_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate source id '2306.11698'"):
        ing.check_source_ids([{"arxiv_id": "2306.11698"}, {"path": "docs/x.md", "id": "2306.11698"}])
    with pytest.raises(ValueError, match="duplicate source id 'notes'"):
        ing.check_source_ids([{"path": "a/notes.md"}, {"path": "b/notes.txt"}])


def test_check_source_ids_rejects_an_entry_with_no_source():
    with pytest.raises(ValueError, match="neither 'arxiv_id' nor 'path'"):
        ing.check_source_ids([{"title": "orphan"}])
