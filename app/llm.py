import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from app import config

load_dotenv()


def get_client():
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=os.environ[config.LLM_API_KEY_ENV],
    )


def chat(messages, client=None):
    client = client or get_client()
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = float(getattr(e, "response", None)
                         and e.response.headers.get("retry-after", 2) or 2)
            time.sleep(wait + 0.5)
    raise RuntimeError("Groq rate limit: exhausted retries")


def format_citations(docs):
    seen, out = set(), []
    for d in docs:
        if d.get("source") == "web":
            tag = f"[web: {d['paper_id']}]"
        else:
            tag = f"[{d['title']}, chunk#{d['chunk_index']}]"
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _context_block(docs):
    return "\n\n".join(
        f"[{d['title']}, chunk#{d['chunk_index']}]\n{d['text']}" for d in docs
    )


def grade(query, docs, client=None):
    msg = [
        {"role": "system", "content":
         "You grade whether retrieved context is sufficient to answer a "
         "question. Reply with exactly one word: SUFFICIENT or WEAK."},
        {"role": "user", "content":
         f"Question: {query}\n\nContext:\n{_context_block(docs)}"},
    ]
    out = chat(msg, client).upper()
    return "weak" if "WEAK" in out else "sufficient"


def rewrite(query, client=None):
    msg = [
        {"role": "system", "content":
         "Rewrite the user's question into a more effective search query for "
         "a research-paper retrieval system. Return only the rewritten query."},
        {"role": "user", "content": query},
    ]
    return chat(msg, client)


def answer(query, docs, client=None):
    if not docs:
        return {"answer": "Not grounded in the corpus.", "citations": []}
    msg = [
        {"role": "system", "content":
         "You are a research assistant. Answer the question using ONLY the "
         "provided context passages, and cite sources inline with the "
         "[title, chunk#N] tags shown. If — and only if — none of the passages "
         "are relevant to the question, reply with exactly: "
         "Not grounded in the corpus."},
        {"role": "user", "content":
         f"Question: {query}\n\nContext:\n{_context_block(docs)}"},
    ]
    text = chat(msg, client)
    return {"answer": text, "citations": format_citations(docs)}
