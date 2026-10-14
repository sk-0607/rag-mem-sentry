import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from app import config

load_dotenv()


def get_client():
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=os.environ[config.LLM_API_KEY_ENV],
    )


_last_call = 0.0


def _pace():
    """Space requests out by config.LLM_MIN_INTERVAL, so a long eval run stays
    under the free tier's per-minute cap instead of bursting into a 429."""
    global _last_call
    interval = config.LLM_MIN_INTERVAL
    if interval:
        gap = time.monotonic() - _last_call
        if gap < interval:
            time.sleep(interval - gap)
    _last_call = time.monotonic()


def _backoff(exc, attempt):
    """Honour a server-supplied retry-after, else back off exponentially.
    A flat delay can exhaust every attempt inside one rate-limit window."""
    resp = getattr(exc, "response", None)
    retry_after = resp.headers.get("retry-after") if resp is not None else None
    if retry_after:
        return float(retry_after)
    return min(2 * 2 ** attempt, 60)


def chat(messages, client=None, model=None):
    client = client or get_client()
    model = model or config.LLM_MODEL
    for attempt in range(6):
        _pace()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            # A 503 "high demand" overload or a dropped connection is as transient
            # as a 429; failing it dropped S2 for a whole batch and marked the rows
            # reduced assurance.
            time.sleep(_backoff(e, attempt) + 0.5)
    raise RuntimeError(f"{model} rate limit: exhausted retries")


def _cite_tag(d):
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


def answer(query, docs, experiences="", client=None):
    content = f"Question: {query}\n\nContext:\n{_context_block(docs)}"
    if experiences:
        # Appended, not interpolated into Context: the experiences block is
        # agent-authored memory, not retrieved corpus text, and keeping it a
        # separate labeled block (format_experiences supplies the label)
        # matches the _context_block convention instead of inventing a new one.
        content += f"\n\n{experiences}"
    msg = [
        {"role": "system", "content":
         "You are a research assistant. Answer the question using ONLY the "
         "provided context passages. Cite sources inline by copying the "
         "bracketed tag shown above a passage verbatim, brackets included. "
         "Give every source its own brackets — write [tag one] [tag two], "
         "never [tag one, tag two]."},
        {"role": "user", "content": content},
    ]
    return {"answer": chat(msg, client), "citations": format_citations(docs)}


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
