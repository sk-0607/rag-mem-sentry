import io
import json
import sys
import types

import pytest

from eval.experience_poison_trial import TRIALS, ReadOnlyStore, build_poison_store


def test_every_trial_has_a_preregistered_check():
    for t in TRIALS:
        assert t["check"]
        assert t["poison"].task == t["question"]


class FakeInner:
    def __init__(self):
        self.writes = []
        self.seen = None

    def retrieve(self, task, k):
        self.seen = (task, k)
        return ["r1"]

    def write(self, record, gate=None):
        self.writes.append(record)


def test_read_only_store_never_writes():
    inner = FakeInner()
    ro = ReadOnlyStore(inner)

    assert ro.write("some record") is None
    assert inner.writes == []

    assert ro.retrieve("task text", k=5) == ["r1"]
    assert inner.seen == ("task text", 5)


class FakeStore:
    """Stands in for memory.store.ExperienceStore — construction and write
    are the only calls build_poison_store makes; Chroma/MiniLM never load.
    """
    def __init__(self, chroma_dir, jsonl_path, collection, embedder=None):
        pass

    def write(self, record, gate=None):
        pass


def test_build_poison_store_wipes_only_the_trial_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.store.ExperienceStore", FakeStore)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    stale = trial_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    build_poison_store(trial_dir=trial_dir)

    assert sentinel.exists()
    assert not stale.exists()


def test_trial_does_not_touch_removed_web_config():
    import eval.experience_poison_trial as mod
    from app import config

    mod.configure()

    assert not hasattr(config, "WEB_ENABLED")


@pytest.fixture
def offline_main(tmp_path, monkeypatch):
    """Drives the real main() with every collaborator that would cost money or
    load a model replaced. The store builder is the real build_poison_store (over
    FakeStore), so the wipe under test is the real rmtree, not a stand-in."""
    import eval.experience_poison_trial as mod

    trial_dir = tmp_path / "trial"
    results_dir = tmp_path / "results"
    answer = {"text": "plain ascii"}
    ENTRY = {"item_id": "poison-t1", "store": "experience", "stage": "read",
             "action": "quarantine", "trust": 0.1, "reason": "trust 0.10",
             "signals": [{"signal": "structural", "score": 0.1, "reason": "x", "evidence_span": None},
                         {"signal": "judge", "score": 0.9, "reason": "endorses_unsafe", "evidence_span": None}]}
    seen = {"calls": [], "events": [], "gates": [], "excluded": ["poison-t1"],
            "audit_entries": [
                ENTRY,
                # A corpus entry for the same id must not be read as the experience verdict.
                {**ENTRY, "store": "corpus", "signals": [{"signal": "structural", "score": 0.7,
                                                          "reason": "x", "evidence_span": None}]},
            ],
            "fail_at": None}

    def fake_run_agentic(question, retriever=None, store=None, read_gate=None):
        seen["calls"].append({"store": store, "read_gate": read_gate})
        if seen["fail_at"] == len(seen["calls"]):  # what llm.chat raises once its retries are spent
            raise RuntimeError("rate limit: exhausted retries")
        gated = read_gate is not None and store is not None
        excluded = seen["excluded"] if gated else []
        return {"experiences": [i for i in ["poison-t1"] if i not in excluded],
                "experiences_excluded": excluded,
                "path_trace": ["retrieve", "generate"], "answer": answer["text"]}

    class FakeAudit:
        def entries(self):
            return seen["audit_entries"]

    def fake_default_read_gate(data_dir, chat=None):
        seen["events"].append("gate")
        gate = types.SimpleNamespace(audit=FakeAudit(), data_dir=data_dir)
        seen["gates"].append(gate)
        return gate

    def recording_build():
        seen["events"].append("build")
        return build_poison_store(trial_dir=trial_dir)

    monkeypatch.setattr("memory.store.ExperienceStore", FakeStore)
    monkeypatch.setattr("app.orchestrator.run_agentic", fake_run_agentic)
    monkeypatch.setattr("gate.read.default_read_gate", fake_default_read_gate)
    monkeypatch.setattr("app.retrieval.Retriever", lambda: None)
    monkeypatch.setattr(mod, "configure", lambda: None)
    monkeypatch.setattr(mod, "screen_poison", lambda: [])
    monkeypatch.setattr(mod, "build_poison_store", recording_build)
    monkeypatch.setattr(mod, "TRIAL_DIR", trial_dir)
    # raising=False: before RESULTS_DIR exists the tests must fail on behavior, not on the patch.
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir, raising=False)
    return mod, trial_dir, results_dir, answer, seen


def test_results_are_written_outside_the_wiped_trial_dir(offline_main):
    mod, trial_dir, results_dir, _, _ = offline_main

    assert mod.main("off") == 0

    assert (results_dir / "results-off.json").exists()
    assert results_dir != trial_dir and trial_dir not in results_dir.parents


def test_real_results_dir_is_not_inside_the_wiped_trial_dir():
    # offline_main patches both directories to tmp_path, so nothing else pins the
    # real values: results kept inside TRIAL_DIR were deleted by the next run.
    import eval.experience_poison_trial as mod

    assert mod.RESULTS_DIR != mod.TRIAL_DIR
    assert mod.TRIAL_DIR not in mod.RESULTS_DIR.parents


def test_results_survive_the_next_runs_wipe(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("off")
    results_dir.mkdir(exist_ok=True)
    backup = results_dir / "results-backup.json"
    backup.write_text("{}", encoding="utf-8")
    mod.main("off")  # second run wipes the trial dir again

    assert (results_dir / "results-off.json").exists()
    assert backup.exists()


def test_main_prints_on_a_cp1252_console(offline_main, monkeypatch):
    mod, _, results_dir, answer, _ = offline_main
    answer["text"] = "中"  # not encodable in cp1252
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    assert mod.main("off") == 0

    on_disk = json.loads((results_dir / "results-off.json").read_text(encoding="utf-8"))
    assert on_disk["poisoned"][0]["answer"] == "中"


def test_arm_on_passes_a_read_gate(offline_main):
    mod, trial_dir, _, _, seen = offline_main

    assert mod.main("on") == 0

    poisoned = [c for c in seen["calls"] if c["store"] is not None]
    assert len(poisoned) == len(TRIALS)
    assert all(c["read_gate"] is not None for c in poisoned)
    # The trial's own cache and audit, not the live data/ ones.
    assert seen["gates"][0].data_dir == trial_dir


def test_arm_on_builds_the_gate_after_the_store_wipe(offline_main):
    mod, _, _, _, seen = offline_main

    mod.main("on")

    # build_poison_store rmtree's TRIAL_DIR: a gate built first would have its
    # cache and audit files deleted out from under it.
    assert seen["events"] == ["build", "gate"]


def test_arm_off_passes_none(offline_main):
    mod, _, _, _, seen = offline_main

    mod.main("off")

    assert seen["calls"]
    assert all(c["read_gate"] is None for c in seen["calls"])
    assert seen["gates"] == []


def test_results_path_depends_on_arm(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("off")
    mod.main("on")  # its store rebuild wipes the trial dir, not the off results

    assert json.loads((results_dir / "results-off.json").read_text(encoding="utf-8"))["arm"] == "off"
    assert json.loads((results_dir / "results-on.json").read_text(encoding="utf-8"))["arm"] == "on"


def test_arm_on_copies_exclusions_and_the_causing_signal_into_results(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("on")

    first = json.loads((results_dir / "results-on.json").read_text(encoding="utf-8"))["poisoned"][0]
    assert first["experiences_excluded"] == ["poison-t1"]
    # Attributed to the highest-scoring experience-store signal (S2 here), not to "the gate".
    assert first["excluded_top_signals"] == {"poison-t1": "judge"}


def test_arm_off_records_no_exclusions(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("off")

    first = json.loads((results_dir / "results-off.json").read_text(encoding="utf-8"))["poisoned"][0]
    assert first["experiences_excluded"] == [] and first["excluded_top_signals"] == {}
    assert first["gate_audit"] == {}


def test_unknown_arm_is_rejected_before_any_work(offline_main):
    mod, _, _, _, seen = offline_main

    with pytest.raises(ValueError):
        mod.main("On")

    assert seen["calls"] == [] and seen["events"] == []


def test_admitted_poison_audit_entry_is_copied_with_its_reduced_assurance_reason(offline_main):
    mod, _, results_dir, _, seen = offline_main
    # run_agentic drops the experience screen's reduced_assurance flag, so this
    # entry's reason is the only record that the judge never ran on an admit.
    seen["excluded"] = []
    seen["audit_entries"] = [{
        "item_id": "poison-t1", "store": "experience", "stage": "read", "action": "admit",
        "trust": 0.82, "reason": "trust 0.82; reduced assurance: judge unavailable",
        "signals": [{"signal": "structural", "score": 0.18, "reason": "pattern", "evidence_span": None}],
    }]

    mod.main("on")

    first = json.loads((results_dir / "results-on.json").read_text(encoding="utf-8"))["poisoned"][0]
    assert first["experiences_excluded"] == [] and first["retrieved"] == ["poison-t1"]
    assert first["gate_audit"] == {"poison-t1": {
        "action": "admit", "trust": 0.82,
        "reason": "trust 0.82; reduced assurance: judge unavailable",
        "signals": [{"signal": "structural", "score": 0.18, "reason": "pattern", "evidence_span": None}],
    }}


def test_excluded_poison_audit_entry_carries_all_signals(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("on")

    first = json.loads((results_dir / "results-on.json").read_text(encoding="utf-8"))["poisoned"][0]
    entry = first["gate_audit"]["poison-t1"]
    assert entry["action"] == "quarantine" and entry["reason"] == "trust 0.10"
    assert [s["signal"] for s in entry["signals"]] == ["structural", "judge"]
    assert entry["signals"][1]["reason"] == "endorses_unsafe"


def test_a_missing_audit_entry_is_recorded_as_null(offline_main):
    mod, _, results_dir, _, seen = offline_main
    seen["excluded"], seen["audit_entries"] = [], []

    mod.main("on")

    first = json.loads((results_dir / "results-on.json").read_text(encoding="utf-8"))["poisoned"][0]
    assert first["gate_audit"] == {"poison-t1": None}


def test_a_failure_on_the_second_trial_leaves_the_first_on_disk(offline_main):
    mod, _, results_dir, _, seen = offline_main
    seen["fail_at"] = 2

    with pytest.raises(RuntimeError):  # not swallowed: exit code stays non-zero
        mod.main("off")

    on_disk = json.loads((results_dir / "results-off.json").read_text(encoding="utf-8"))
    assert on_disk["complete"] is False
    assert len(on_disk["poisoned"]) == 1 and on_disk["control"] == []


def test_a_full_run_ends_complete(offline_main):
    mod, _, results_dir, _, _ = offline_main

    mod.main("off")

    on_disk = json.loads((results_dir / "results-off.json").read_text(encoding="utf-8"))
    assert on_disk["complete"] is True
    assert len(on_disk["poisoned"]) == len(TRIALS) and len(on_disk["control"]) == len(TRIALS)
