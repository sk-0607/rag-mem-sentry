from app.chunking import chunk_text

def test_chunks_have_metadata_and_ids():
    text = "Intro line.\n\n" + ("word " * 500)
    chunks = chunk_text(text, paper_id="1234.5678", title="Sample")
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c["paper_id"] == "1234.5678"
        assert c["title"] == "Sample"
        assert c["chunk_index"] == i
        assert c["id"] == f"1234.5678::{i}"
        assert c["text"].strip()

def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, "p", "t", target_words=100, overlap_words=20)
    # consecutive chunks should share tail/head words (overlap)
    first_tail = chunks[0]["text"].split()[-20:]
    second_head = chunks[1]["text"].split()[:20]
    assert first_tail == second_head

def test_empty_text_returns_empty():
    assert chunk_text("   ", "p", "t") == []
