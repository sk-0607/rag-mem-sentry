import json

from eval import corroborate_fpr
from eval.corroborate_fpr import holdout_split, summarize
from gate.policy import DEFAULT_POLICY
from gate.read import ReadResult
from gate.score import aggregate
from gate.types import SignalResult

import pytest

SOURCES = ["src-a", "src-b", "src-c", "src-d", "src-e"]


def make_chunks(per_source=5):
    chunks = [{"id": f"{s}-{i}", "text": f"text {s} {i}", "paper_id": s, "trusted": True}
              for s in SOURCES for i in range(per_source)]
    chunks += [{"id": f"src-f-{i}", "text": "untrusted", "paper_id": "src-f", "trusted": False}
               for i in range(per_source)]
    return chunks


def test_sources_sampled_from_trusted_sources_only():
    for seed in range(10):
        _, _, picked = holdout_split(make_chunks(), 2, 3, seed=seed)
        assert "src-f" not in picked and len(picked) == 2


def test_split_is_reproducible_for_a_seed():
    chunks = make_chunks()
    assert holdout_split(chunks, 2, 3, seed=1) == holdout_split(chunks, 2, 3, seed=1)
    assert len({tuple(holdout_split(chunks, 2, 3, seed=s)[2]) for s in range(10)}) > 1


def test_reference_excludes_every_held_out_source():
    ref, held, picked = holdout_split(make_chunks(), 2, 4)
    assert not {c["paper_id"] for c in ref} & set(picked)
    assert {c["paper_id"] for c in held} <= set(picked)
    # untrusted src-f chunks are not reference either
    assert all(c["trusted"] is True for c in ref)


def test_held_out_chunks_are_untrusted_copies():
    chunks = make_chunks()
    _, held, _ = holdout_split(chunks, 2, 4)
    assert held and all(c["trusted"] is False for c in held)
    assert all(c["trusted"] is True for c in chunks if c["paper_id"] != "src-f")


def test_sample_size_capped_by_available_chunks():
    _, held, picked = holdout_split(make_chunks(per_source=3), 2, 40)
    assert len(held) == 6


def test_too_few_trusted_sources_raises():
    with pytest.raises(ValueError):
        holdout_split(make_chunks(), len(SOURCES), 3)


def row(action="admit", top="corroborate", reason=None, reduced=False):
    signals = [{"signal": "corroborate", "score": 0.5, "reason": reason}] if reason else []
    return {"action": action, "top_signal": top, "signals": signals, "reduced_assurance": reduced}


def test_summarize_counts_s3_topped_non_admits():
    s = summarize([row("quarantine", "corroborate", "contradicted"),
                   row("quarantine", "structural", "contradicted"),
                   row("admit", "corroborate", "supported"),
                   row("admit", "corroborate", "supported")])
    assert s["total"] == 4 and s["non_admit"] == 2
    assert s["s3_checked"] == 4 and s["s3_topped_non_admit"] == 1
    assert s["s3_fpr"] == 0.25
    assert s["reasons"] == {"contradicted": 2, "supported": 2}


def test_summarize_separates_not_checked_reasons():
    s = summarize([row(reason=None, top="structural"), row(reason="no_claims"),
                   row(reason="not_covered"), row(reason="not_covered")])
    assert s["s3_not_checked"] == {"absent": 1, "no_claims": 1, "not_covered": 2,
                                  "verbatim_trusted": 0}
    assert s["s3_checked"] == 0


def test_summarize_counts_unparseable_as_checked():
    # An unparseable reply quarantines, so it is a possible false positive and
    # belongs in the denominator, not under "absent".
    s = summarize([row("quarantine", "corroborate", "corroborate reply could not be parsed: x"),
                   row("admit", "corroborate", "supported")])
    assert s["unparseable"] == 1 and s["s3_checked"] == 2
    assert s["s3_not_checked"]["absent"] == 0
    assert s["s3_topped_non_admit"] == 1 and s["s3_fpr"] == 0.5


def test_summarize_counts_verbatim_trusted_under_its_own_key():
    s = summarize([row(reason="verbatim_trusted"), row(reason=None, top="structural")])
    assert s["s3_not_checked"] == {"absent": 1, "no_claims": 0, "not_covered": 0, "verbatim_trusted": 1}
    assert s["s3_checked"] == 0


def test_summarize_fpr_is_none_when_nothing_checked():
    assert summarize([row(reason="no_claims")])["s3_fpr"] is None


def test_summarize_excludes_reduced_assurance_rows():
    s = summarize([row("quarantine", reason="contradicted", reduced=True),
                   row("admit", reason="supported")])
    assert s["reduced_assurance"] == 1 and s["total"] == 1
    assert s["non_admit"] == 0 and s["s3_checked"] == 1


def test_cli_refuses_an_empty_trusted_set(tmp_path, capsys):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({"id": "x", "text": "t", "paper_id": "src-a", "trusted": False}) + "\n")
    assert corroborate_fpr.main([], chunks_path=path, out_dir=tmp_path / "out") != 0
    assert "Task 43" in capsys.readouterr().err


def test_cli_gate_uses_the_reduced_reference(tmp_path, monkeypatch):
    chunks = make_chunks()
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in chunks))
    seen = {}

    class Gate:
        def screen_batch(self, items, store):
            seen["items"] = items
            v = DEFAULT_POLICY.decide(aggregate([SignalResult("structural", 0.0, "r")]), store)
            return ReadResult([v for _ in items], False)

    def fake_default_read_gate(data_dir, chat=None, reference=None):
        seen["ref_len"] = len(reference)
        return Gate()

    monkeypatch.setattr(corroborate_fpr, "default_read_gate", fake_default_read_gate)
    monkeypatch.setenv("HOLDOUT_SOURCES", "2")
    monkeypatch.setenv("HOLDOUT_CHUNKS", "4")
    assert corroborate_fpr.main([], chunks_path=path, out_dir=tmp_path / "out") == 0
    assert seen["ref_len"] == 15
    assert all(i.metadata == {"trusted": False} for i in seen["items"])
    out = json.loads((tmp_path / "out" / "results.json").read_text())
    assert len(out["rows"]) == 4 and out["rows"][0]["paper_id"] in out["held_out_sources"]
    assert out["summary"]["total"] == 4
