import json

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from app import config


def rrf_merge(ranked_lists, k=60):
    scores = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = client.get_collection(config.COLLECTION)

        self.chunks = {}
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self.chunks[c["id"]] = c
        corpus_ids = list(self.chunks)
        self.bm25_ids = corpus_ids
        self.bm25 = BM25Okapi(
            [self.chunks[i]["text"].lower().split() for i in corpus_ids]
        )

    def _dense_ids(self, query, n):
        emb = self.embedder.encode([query]).tolist()
        res = self.col.query(query_embeddings=emb, n_results=n)
        return res["ids"][0]

    def _sparse_ids(self, query, n):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25_ids[i] for i in ranked[:n]]

    def retrieve(self, query, k=config.TOP_K):
        dense = self._dense_ids(query, config.DENSE_N)
        sparse = self._sparse_ids(query, config.SPARSE_N)
        fused = rrf_merge([dense, sparse])[:config.FUSE_M]
        cand = [self.chunks[i] for i in fused if i in self.chunks]
        if not cand:
            return []
        pairs = [(query, c["text"]) for c in cand]
        rerank_scores = self.reranker.predict(pairs)
        for c, s in zip(cand, rerank_scores):
            c["score"] = float(s)
        cand.sort(key=lambda c: c["score"], reverse=True)
        return cand[:k]
