import io
import json
import sys

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

    def fake_run_agentic(question, retriever=None, store=None):
        return {"experiences": ["poison-t1"], "path_trace": ["retrieve", "generate"],
                "answer": answer["text"]}

    monkeypatch.setattr("memory.store.ExperienceStore", FakeStore)
    monkeypatch.setattr("app.orchestrator.run_agentic", fake_run_agentic)
    monkeypatch.setattr("app.retrieval.Retriever", lambda: None)
    monkeypatch.setattr(mod, "configure", lambda: None)
    monkeypatch.setattr(mod, "screen_poison", lambda: [])
    monkeypatch.setattr(mod, "build_poison_store", lambda: build_poison_store(trial_dir=trial_dir))
    monkeypatch.setattr(mod, "TRIAL_DIR", trial_dir)
    # raising=False: before RESULTS_DIR exists the tests must fail on behavior, not on the patch.
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir, raising=False)
    return mod, trial_dir, results_dir, answer


def test_results_are_written_outside_the_wiped_trial_dir(offline_main):
    mod, trial_dir, results_dir, _ = offline_main

    assert mod.main() == 0

    assert (results_dir / "results.json").exists()
    assert results_dir != trial_dir and trial_dir not in results_dir.parents


def test_results_survive_the_next_runs_wipe(offline_main):
    mod, _, results_dir, _ = offline_main

    mod.main()
    results_dir.mkdir(exist_ok=True)
    backup = results_dir / "results-backup.json"
    backup.write_text("{}", encoding="utf-8")
    mod.main()  # second run wipes the trial dir again

    assert (results_dir / "results.json").exists()
    assert backup.exists()


def test_main_prints_on_a_cp1252_console(offline_main, monkeypatch):
    mod, _, results_dir, answer = offline_main
    answer["text"] = "\u4e2d"  # not encodable in cp1252
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    assert mod.main() == 0

    on_disk = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert on_disk["poisoned"][0]["answer"] == "\u4e2d"
