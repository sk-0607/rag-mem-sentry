import json
import os
import sys
import types
from pathlib import Path

# ragas 0.4.x hard-imports `langchain_community.chat_models.vertexai`, a submodule
# removed in langchain-community 0.4.x (Vertex AI moved to langchain-google-vertexai).
# Our judge is Gemini (ChatOpenAI); we never touch Vertex, so register a stub for the
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


def results_path(env):
    # Separate file per arm so a gate-on run cannot overwrite the gate-off baseline.
    return "eval/results-gate-on.md" if env.get("EVAL_READ_GATE") == "1" else "eval/results.md"


def make_runner(env, run_agentic, gate_factory):
    # Built once, outside the closure: one gate means one verdict cache shared
    # across the whole run, as production would have.
    read_gate = gate_factory() if env.get("EVAL_READ_GATE") == "1" else None
    return lambda question: run_agentic(question, read_gate=read_gate)


def with_traces(runner):
    # build_records drops path_trace (RAGAS has no use for it), but the write-up
    # must count sub-queries whose gate admitted 0 of 5 chunks. Wrapping the
    # runner keeps build_records' record shape untouched.
    traces = []

    def traced(question):
        res = runner(question)
        traces.append({"question": question, "path_trace": res.get("path_trace", [])})
        return res

    return traced, traces


def nan_counts(df, metric_cols):
    # df[c].mean() silently skips NaN rows. A NaN means a daily cap was hit
    # mid-run, so the run is incomplete, not something to average over. A metric
    # ragas dropped entirely counts as missing on every row, not as zero NaNs.
    return {c: int(df[c].isna().sum()) if c in df.columns else len(df) for c in metric_cols}


def run_record_path(env):
    arm = "on" if env.get("EVAL_READ_GATE") == "1" else "off"
    return f"data/phase3_ragas/run-{arm}.json"


def write_run_record(path, traces, scores, nans):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"complete": not any(nans.values()), "scores": scores,
              "nan_counts": nans, "traces": traces}
    open(path, "w", encoding="utf-8").write(json.dumps(record, indent=2))


def write_results(scores, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = ["# RAGAS Evaluation Results\n", "| Metric | Score |", "|---|---|"]
    for k, v in scores.items():
        lines.append(f"| {k} | {v:.2f} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    from app.orchestrator import run_agentic
    from gate.read import default_read_gate
    qa = json.load(open("eval/qa_set.json", encoding="utf-8"))
    # Gemini free-tier is rate-limited; EVAL_N caps the run size for a first pass.
    n = int(os.getenv("EVAL_N", str(len(qa))))
    qa = qa[:n]
    # Generating the answers is the bursty phase (each question fans out into
    # several sub-queries); pace it under the 15 req/min free-tier cap. The
    # judge phase below is already throttled by RunConfig's tenacity backoff.
    config.LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4"))
    # The real data dir: the shared verdict cache, as production would use it.
    run, traces = with_traces(make_runner(
        os.environ, run_agentic,
        lambda: default_read_gate(Path(config.EXPERIENCES_PATH).parent)))
    records = build_records(qa, run)
    ds = Dataset.from_list(records)

    judge = ChatOpenAI(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL,
                       api_key=os.environ[config.LLM_API_KEY_ENV], temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

    # The endpoint allows only n=1 per request, so drop answer_relevancy's
    # multi-sample strictness; serialize jobs (max_workers=1) to respect the
    # per-minute limits and avoid the heavy faithfulness metric timing out
    # under parallel bursts.
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
    nans = nan_counts(df, metric_cols)
    write_run_record(run_record_path(os.environ), traces, scores, nans)
    print(scores)
    print("NaN rows per metric:", nans)
    if any(nans.values()):
        # No scorecard for a capped run: its means skip the NaN rows, and a
        # results file is what gets pasted into the README.
        print("INCOMPLETE: NaN rows mean a daily cap was hit; rerun, do not average over them.")
    else:
        write_results(scores, results_path(os.environ))


if __name__ == "__main__":
    main()
