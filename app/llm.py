import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from app import config

load_dotenv()

REFUSAL = "Not grounded in the corpus."


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
    raise RuntimeError(f"{config.LLM_MODEL} rate limit: exhausted retries")


def _is_refusal(text):
    # the model occasionally trails a stray citation tag after the refusal
    return text.strip().startswith(REFUSAL)


def _cite_tag(d):
    if d.get("source") == "web":
        return f"[web: {d['paper_id']}]"
    return f"[{d['title']}, chunk#{d['chunk_index']}]"


def format_citations(docs):
    seen, out = set(), []
    for d in docs:
        tag = _cite_tag(d)
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _context_block(docs):
    return "\n\n".join(f"{_cite_tag(d)}\n{d['text']}" for d in docs)


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
        return {"answer": REFUSAL, "citations": []}
    msg = [
        {"role": "system", "content":
         "You are a research assistant. Answer the question using ONLY the "
         "provided context passages. Cite sources inline by copying the "
         "bracketed tag shown above a passage verbatim, brackets included. "
         "Give every source its own brackets — write [tag one] [tag two], "
         "never [tag one, tag two]. If — and only if — none of the passages "
         "are relevant to the question, reply with exactly: "
         "Not grounded in the corpus."},
        {"role": "user", "content":
         f"Question: {query}\n\nContext:\n{_context_block(docs)}"},
    ]
    text = chat(msg, client)
    if _is_refusal(text):
        return {"answer": REFUSAL, "citations": []}
    return {"answer": text, "citations": format_citations(docs)}


def decompose(question, client=None):
    msg = [
        {"role": "system", "content":
         "Decompose the user's question into at most 3 standalone sub-questions "
         "needed to answer it. If it is already atomic, return it unchanged. "
         "Output one sub-question per line, no numbering or bullets."},
        {"role": "user", "content": question},
    ]
    out = chat(msg, client)
    lines = [ln.strip(" -*\t").strip() for ln in out.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return [question]
    return lines[:config.MAX_SUBQUERIES]


def synthesize(question, sub_results, client=None):
    if len(sub_results) == 1:
        r = sub_results[0]
        return {"answer": r["answer"], "citations": r["citations"]}
    if all(r["answer"] == REFUSAL for r in sub_results):
        return {"answer": REFUSAL, "citations": []}

    seen, cites = set(), []
    for r in sub_results:
        for c in r["citations"]:
            if c not in seen:
                seen.add(c)
                cites.append(c)

    blocks = "\n\n".join(
        f"Sub-question: {r['sub_query']}\nAnswer: {r['answer']}\n"
        f"Citations: {', '.join(r['citations'])}" for r in sub_results
    )
    msg = [
        {"role": "system", "content":
         "Combine the sub-question answers into one coherent answer to the "
         "original question. Preserve the citation tags exactly as given. Do "
         "not add facts beyond the provided answers."},
        {"role": "user", "content": f"Original question: {question}\n\n{blocks}"},
    ]
    return {"answer": chat(msg, client), "citations": cites}
