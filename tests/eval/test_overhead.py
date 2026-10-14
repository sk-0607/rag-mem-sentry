import json
import types
from pathlib import Path

import pytest

from app import config
from eval import overhead
from eval.overhead import count_calls, measure, run_arms


class FakeReference:
    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n

    def fingerprint(self):
        return "ref-fp"


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    # main() checks the real freeze record and builds the real reference; the
    # tests run as if the check were clean and the trusted set non-empty.
    monkeypatch.setattr(overhead, "require_frozen", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(overhead, "default_reference", lambda d: FakeReference(), raising=False)
    monkeypatch.setattr("eval.attack_screen.check", lambda *a, **k: [])


def fake_llm(log=None):
    def chat(messages, client=None, model=None):
        if log is not None:
            log.append(messages)
        return "ok"
    return types.SimpleNamespace(chat=chat)


def test_count_calls_counts_and_restores():
    llm = fake_llm()
    original = llm.chat
    with count_calls(llm) as counter:
        llm.chat([])
        llm.chat([])
    assert counter["chat"] == 2
    assert llm.chat is original


def test_count_calls_restores_on_exception():
    llm = fake_llm()
    original = llm.chat
    with pytest.raises(RuntimeError):
        with count_calls(llm):
            raise RuntimeError("quota")
    assert llm.chat is original


def test_measure_divides_by_question_count():
    llm = fake_llm()

    def arm(question):
        for _ in range(3):
            llm.chat([])

    result = measure(["q1", "q2"], arm, llm_module=llm)
    assert result["questions"] == 2
    assert result["llm_calls"] == 6
    assert result["calls_per_question"] == 3
    assert result["seconds_per_question"] == result["total_seconds"] / 2


class Harness:
    """Fake run_agentic and gate factory. The factory records the data dir it
    was handed and what that dir held at construction time; a gated run drops
    a cache file there and makes one extra chat call, like a judge would."""

    def __init__(self, llm):
        self.llm = llm
        self.built = []  # (data_dir, listing at construction)

    def gate_factory(self, data_dir):
        self.built.append((Path(data_dir), sorted(p.name for p in Path(data_dir).iterdir())))
        return Path(data_dir)

    def run_agentic(self, question, read_gate=None):
        self.llm.chat([])
        if read_gate is not None:
            self.llm.chat([])  # the judge
            (read_gate / "gate_cache.jsonl").write_text("x", encoding="utf-8")
        return {"answer": "a"}


def test_cold_arm_uses_empty_data_dir():
    llm = fake_llm()
    h = Harness(llm)
    arms = run_arms(["q1", "q2"], h.run_agentic, h.gate_factory, llm_module=llm)
    cold_dir, listing_at_build = h.built[0]
    assert listing_at_build == []
    assert cold_dir != Path("data")
    assert arms["on-cold"]["llm_calls"] > arms["off"]["llm_calls"]


def test_warm_arm_reuses_cold_arm_data_dir():
    llm = fake_llm()
    h = Harness(llm)
    run_arms(["q1"], h.run_agentic, h.gate_factory, llm_module=llm)
    (cold_dir, _), (warm_dir, warm_listing) = h.built
    assert warm_dir == cold_dir
    # Built after the cold pass, so the warm gate loads what the cold one wrote.
    assert warm_listing == ["gate_cache.jsonl"]


def test_run_arms_cleans_up_the_temp_dir():
    llm = fake_llm()
    h = Harness(llm)
    run_arms(["q1"], h.run_agentic, h.gate_factory, llm_module=llm)
    assert not h.built[0][0].exists()


def test_run_arms_fails_loudly_when_judge_calls_read_zero():
    llm = fake_llm()
    seen = []

    def run_agentic(question, read_gate=None):
        llm.chat([])  # same call count gated or not: the judge was never counted
        return {}

    with pytest.raises(SystemExit, match="judge"):
        run_arms(["q1"], run_agentic, lambda d: object(), llm_module=llm, on_arm=seen.append)
    assert "off" in seen[-1] and "on-cold" in seen[-1]  # cold arm saved before the raise
    assert "on-warm" not in seen[-1]


def test_main_persists_after_each_arm_and_marks_complete(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(overhead, "RESULTS_DIR", tmp_path / "out")
    monkeypatch.setenv("OVERHEAD_N", "2")
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)  # main() overwrites it; restore after
    llm = fake_llm()
    h = Harness(llm)
    snapshots = []

    real_save = overhead.save_results

    def spy(results):
        real_save(results)
        snapshots.append(json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8")))

    monkeypatch.setattr(overhead, "save_results", spy)
    rc = overhead.main(run_agentic=h.run_agentic, gate_factory=h.gate_factory, llm_module=llm,
                       questions=["q1", "q2", "q3"])
    assert rc == 0
    assert [s["complete"] for s in snapshots] == [False, False, True]
    final = snapshots[-1]
    assert final["overhead_n"] == 2
    assert set(final["arms"]) == {"off", "on-cold", "on-warm"}
    assert "model" in final and "llm_min_interval" in final
    assert json.loads(capsys.readouterr().out)["complete"] is True


def main_with_default_gate(monkeypatch, tmp_path, reference=None):
    """main() with its own gate factory: default_read_gate and default_reference
    are the module's, so what main hands them is observable."""
    monkeypatch.setattr(overhead, "RESULTS_DIR", tmp_path / "out")
    monkeypatch.setenv("OVERHEAD_N", "1")
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL", 0)
    reference = reference if reference is not None else FakeReference()
    seen = {"ref_dirs": [], "gates": []}

    def fake_reference(data_dir):
        seen["ref_dirs"].append(Path(data_dir))
        return reference

    def fake_gate(data_dir, chat=None, reference=None):
        seen["gates"].append((Path(data_dir), sorted(p.name for p in Path(data_dir).iterdir()), reference))
        return Path(data_dir)

    monkeypatch.setattr(overhead, "default_reference", fake_reference)
    monkeypatch.setattr(overhead, "default_read_gate", fake_gate, raising=False)
    log = []
    llm = fake_llm(log)
    llm.log = log
    h = Harness(llm)
    rc = overhead.main(run_agentic=h.run_agentic, llm_module=llm, questions=["q1"])
    return rc, seen, reference, llm


def test_overhead_refuses_unclean_freeze_before_building_gate(tmp_path, monkeypatch, capsys):
    from eval.freeze_gate import FreezeError

    def unclean(*a, **k):
        raise FreezeError("gate changed since freeze")

    monkeypatch.setattr(overhead, "require_frozen", unclean)
    rc, seen, _, llm = main_with_default_gate(monkeypatch, tmp_path)
    assert rc != 0
    assert seen["gates"] == [] and llm.log == []
    assert not (tmp_path / "out").exists()
    assert "gate changed" in capsys.readouterr().err


def test_overhead_records_freeze_field(tmp_path, monkeypatch):
    from eval.freeze_gate import FROZEN_GATE
    rc, _, _, _ = main_with_default_gate(monkeypatch, tmp_path)
    assert rc == 0
    assert FROZEN_GATE.exists()
    on_disk = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert on_disk["freeze"] == {"record": FROZEN_GATE.as_posix(), "differences": []}


def test_overhead_passes_real_reference(tmp_path, monkeypatch):
    rc, seen, reference, _ = main_with_default_gate(monkeypatch, tmp_path)
    assert rc == 0
    assert len(seen["gates"]) == 2
    assert all(ref is reference for _, _, ref in seen["gates"])
    assert seen["ref_dirs"] == [Path(config.CHUNKS_PATH).parent]


def test_overhead_refuses_empty_reference(tmp_path, monkeypatch):
    rc, seen, _, llm = main_with_default_gate(monkeypatch, tmp_path, reference=FakeReference(0))
    assert rc != 0
    assert seen["gates"] == [] and llm.log == []


def test_cold_arm_cache_is_empty_with_real_reference(tmp_path, monkeypatch):
    rc, seen, reference, _ = main_with_default_gate(monkeypatch, tmp_path)
    assert rc == 0
    cold_dir, listing, ref = seen["gates"][0]
    assert listing == []
    assert cold_dir != Path(config.CHUNKS_PATH).parent
    assert ref is reference
