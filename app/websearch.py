from ddgs import DDGS

from app import config


def _domain(url):
    if "://" in url:
        return url.split("://", 1)[1].split("/", 1)[0]
    return url.split("/", 1)[0]


def web_search(query, k=config.WEB_K):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=k))
        docs = []
        for r in results:
            url = r.get("href") or r.get("url", "")
            docs.append({
                "id": url,
                "text": r.get("body", ""),
                "paper_id": _domain(url),
                "title": r.get("title", ""),
                "chunk_index": 0,
                "source": "web",
                "url": url,
                "score": 0.0,
            })
        return docs
    except Exception:
        return []
