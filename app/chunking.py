def chunk_text(text, paper_id, title, target_words=220, overlap_words=40):
    words = text.split()
    if not words:
        return []
    step = max(1, target_words - overlap_words)
    chunks = []
    idx = 0
    for start in range(0, len(words), step):
        window = words[start:start + target_words]
        if not window:
            break
        chunks.append({
            "id": f"{paper_id}::{idx}",
            "text": " ".join(window),
            "paper_id": paper_id,
            "title": title,
            "chunk_index": idx,
        })
        idx += 1
        if start + target_words >= len(words):
            break
    return chunks
