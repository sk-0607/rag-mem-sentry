"""Build the NQ corpus: 50 PoisonedRAG questions, their gold passages, seeded distractors.

The final corpus is a BEIR NQ subset (owner decision 2026-10-14) so PoisonedRAG's
class-2 poison targets facts the trusted set actually holds.

    python -m ingest.build_nq <corpus-src>   # reads <src>/beir-nq/nq and <src>/PoisonedRAG;
                                             # writes data/nq_docs/, data/nq_selection.jsonl,
                                             # ingest/papers.json. No LLM, no network.
    python -m ingest.build_nq <corpus-src> holdout   # data/nq_holdout.jsonl: clean sibling
                                                     # passages of gold articles, for S3's FPR
"""
import csv
import json
import random
import sys
from pathlib import Path

N_QUESTIONS = 50
N_DISTRACTORS = 2000
MIN_WORDS = 20
UNTRUSTED_FRAC = 0.1
SEED = 0
DOCS_DIR = Path("data/nq_docs")


def pick_questions(prag: dict, qrels: dict, n: int) -> list[dict]:
    # File order. A false answer inside its own question fails the attack-corpus
    # validator, and a question with no gold passage has no fact for S3 to hold.
    picked = []
    for qid, e in prag.items():
        if qid in qrels and e["incorrect answer"].casefold() not in e["question"].casefold():
            picked.append({"qid": qid, "question": e["question"], "correct": e["correct answer"],
                           "incorrect": e["incorrect answer"], "gold": qrels[qid]})
            if len(picked) == n:
                break
    return picked


def group_docs(passages: list[dict], min_words: int) -> list[dict]:
    # One document per Wikipedia title. Short distractors (some are one word) are
    # noise; a short gold passage still carries a question's true fact, so it stays.
    by_title: dict[str, dict] = {}
    for p in passages:
        if not p["gold"] and len(p["text"].split()) < min_words:
            continue
        d = by_title.setdefault(p["title"], {"title": p["title"], "parts": [], "gold": False})
        d["parts"].append(p["text"])
        d["gold"] = d["gold"] or p["gold"]
    return [{"title": d["title"], "text": "\n\n".join(d["parts"]), "gold": d["gold"]}
            for d in by_title.values()]


def assign_trust(docs: list[dict], untrusted_frac: float, seed: int) -> list[dict]:
    # Held out from distractors only: an untrusted gold document would leave S3
    # no trusted fact to contradict that question's poison with.
    pool = [i for i, d in enumerate(docs) if not d["gold"]]
    held = set(random.Random(seed).sample(pool, round(untrusted_frac * len(pool))))
    return [{"id": f"nq-{i:04d}", "path": (DOCS_DIR / f"nq-{i:04d}.txt").as_posix(),
             "title": d["title"], "trusted": i not in held} for i, d in enumerate(docs)]


def siblings(corpus_path: Path, gold_ids: set, indexed_texts: set, min_words: int) -> list[dict]:
    # Clean held-out text the trusted set covers: other passages of a gold
    # article. A random distractor is almost never covered, so S3 checked 0 / 40
    # of those (README, "Final corpus: BEIR NQ"). Two passes: titles, then rows.
    titles = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in gold_ids:
                titles.add(doc["title"])
    out = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if (doc["title"] in titles and doc["_id"] not in gold_ids
                    and doc["text"] not in indexed_texts and len(doc["text"].split()) >= min_words):
                out.append({"id": f"nqh-{doc['_id']}", "text": doc["text"], "paper_id": doc["title"]})
    return out


def write_holdout(src: Path) -> int:
    gold = {c for line in Path("data/nq_selection.jsonl").read_text(encoding="utf-8").splitlines()
            for c in json.loads(line)["gold"]}
    indexed = {p for f in DOCS_DIR.glob("*.txt") for p in f.read_text(encoding="utf-8").split("\n\n")}
    rows = siblings(src / "beir-nq/nq/corpus.jsonl", gold, indexed, MIN_WORDS)
    Path("data/nq_holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                             encoding="utf-8")
    print(f"{len(rows)} held-out sibling passages from {len({r['paper_id'] for r in rows})} gold articles")
    return 0


def stage(src: Path) -> tuple[list[dict], list[dict]]:
    prag = json.load(open(src / "PoisonedRAG/results/adv_targeted_results/nq.json", encoding="utf-8"))
    qrels: dict[str, list[str]] = {}
    with open(src / "beir-nq/nq/qrels/test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
    picked = pick_questions(prag, qrels, N_QUESTIONS)
    gold = {c for p in picked for c in p["gold"]}

    # One pass over the 2.68M-line corpus: keep gold, reservoir-sample distractors.
    rng = random.Random(SEED)
    kept, reservoir, seen = [], [], 0
    with open(src / "beir-nq/nq/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in gold:
                kept.append({**doc, "gold": True})
                continue
            seen += 1
            if len(reservoir) < N_DISTRACTORS:
                reservoir.append({**doc, "gold": False})
            elif (j := rng.randrange(seen)) < N_DISTRACTORS:
                reservoir[j] = {**doc, "gold": False}
    return picked, kept + reservoir


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "holdout":
        return write_holdout(Path(argv[0]))
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    picked, passages = stage(Path(argv[0]))
    docs = group_docs(passages, MIN_WORDS)
    entries = assign_trust(docs, UNTRUSTED_FRAC, SEED)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for e, d in zip(entries, docs):
        Path(e["path"]).write_text(d["text"], encoding="utf-8")
    Path("data/nq_selection.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in picked), encoding="utf-8")
    Path("ingest/papers.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    untrusted = sum(not e["trusted"] for e in entries)
    print(f"{len(picked)} questions, {len(docs)} documents "
          f"({sum(d['gold'] for d in docs)} gold), {untrusted} untrusted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
