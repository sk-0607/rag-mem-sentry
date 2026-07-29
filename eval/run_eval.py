import json
import os
import sys
import types

# ragas 0.4.x hard-imports `langchain_community.chat_models.vertexai`, a submodule
# removed in langchain-community 0.4.x (Vertex AI moved to langchain-google-vertexai).
# Our judge is Groq (ChatOpenAI); we never touch Vertex, so register a stub for the
# single name ragas references, before importing ragas.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vx = types.ModuleType("langchain_community.chat_models.vertexai")
    _vx.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vx

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig

from app import config

load_dotenv()


def build_records(qa_pairs, runner):
    records = []
    for qa in qa_pairs:
        res = runner(qa["question"])
        records.append({
            "question": qa["question"],
            "answer": res["answer"],
            "contexts": res.get("contexts", []),
            "ground_truth": qa["ground_truth"],
        })
    return records


def write_results(scores, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = ["# RAGAS Evaluation Results\n", "| Metric | Score |", "|---|---|"]
    for k, v in scores.items():
        lines.append(f"| {k} | {v:.2f} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    from app.orchestrator import run_agentic as run
    qa = json.load(open("eval/qa_set.json", encoding="utf-8"))
    # Groq free-tier is TPM-limited; EVAL_N caps the run size for a first pass.
    n = int(os.getenv("EVAL_N", str(len(qa))))
    qa = qa[:n]
    records = build_records(qa, run)
    ds = Dataset.from_list(records)

    judge = ChatOpenAI(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL,
                       api_key=os.environ[config.LLM_API_KEY_ENV], temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

    # Groq allows only n=1 per request, so drop answer_relevancy's multi-sample
    # strictness; serialize jobs (max_workers=1) to respect TPM and avoid the
    # heavy faithfulness metric timing out under parallel bursts.
    answer_relevancy.strictness = 1
    run_config = RunConfig(timeout=900, max_workers=1)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    df = result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    scores = {c: float(df[c].mean()) for c in metric_cols if c in df.columns}
    write_results(scores, "eval/results.md")
    print(scores)


if __name__ == "__main__":
    main()
