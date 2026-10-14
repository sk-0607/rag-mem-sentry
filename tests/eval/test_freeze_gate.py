import dataclasses
import json
import subprocess

import pytest

from app import config
from eval import freeze_gate
from eval.freeze_gate import FreezeError, check, diff, gate_state, record, require_frozen
from gate.policy import DEFAULT_POLICY, StoreThresholds


class FakeReference:
    def __init__(self, fp="fp-1"):
        self.fp = fp

    def fingerprint(self):
        return self.fp

    def __len__(self):
        return 1


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "gate" / "signals").mkdir(parents=True)
    (root / "gate" / "policy.py").write_text("A = 1\n", encoding="utf-8")
    (root / "gate" / "signals" / "one.py").write_text("B = 2\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "init")
    ref = FakeReference()
    monkeypatch.setattr(freeze_gate, "default_reference", lambda data_dir: ref)
    monkeypatch.setenv("JUDGE_MODEL", "judge-x")
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-x")
    return root, ref, tmp_path / "frozen.json"


def _record(repo):
    root, _, path = repo
    return record(path, root / "data", "2026-01-01", root=root)


def _check(repo):
    root, _, path = repo
    return check(path, root / "data", root=root)


def test_record_then_check_is_clean(repo):
    _record(repo)
    assert _check(repo) == []


def test_changed_gate_file_is_reported(repo):
    _record(repo)
    (repo[0] / "gate" / "policy.py").write_text("A = 2\n", encoding="utf-8")
    assert _check(repo) == ["gate file changed: gate/policy.py"]


def test_new_untracked_gate_file_is_reported(repo):
    _record(repo)
    (repo[0] / "gate" / "signals" / "extra.py").write_text("C = 3\n", encoding="utf-8")
    assert _check(repo) == ["gate file added: gate/signals/extra.py"]


def test_ignored_gate_file_is_not_reported(repo):
    _record(repo)
    (repo[0] / "gate" / "__pycache__").mkdir()
    (repo[0] / "gate" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert _check(repo) == []


def test_removed_gate_file_is_reported(repo):
    _record(repo)
    (repo[0] / "gate" / "signals" / "one.py").unlink()
    assert _check(repo) == ["gate file removed: gate/signals/one.py"]


def test_knob_change_is_reported(repo, monkeypatch):
    _record(repo)
    monkeypatch.setattr(config, "S3_REF_K", config.S3_REF_K + 1)
    assert _check(repo) == [f"knob S3_REF_K: {config.S3_REF_K - 1} -> {config.S3_REF_K}"]


def test_judge_model_change_is_reported(repo, monkeypatch):
    _record(repo)
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-y")
    assert _check(repo) == ["judge_model: judge-x -> judge-y"]


def test_policy_value_change_is_reported(repo, monkeypatch):
    _record(repo)
    moved = dataclasses.replace(DEFAULT_POLICY, thresholds={
        **DEFAULT_POLICY.thresholds, "corpus": StoreThresholds(block_below=0.15, quarantine_below=0.60)})
    monkeypatch.setattr(freeze_gate, "DEFAULT_POLICY", moved)
    assert _check(repo) == ["policy thresholds.corpus.block_below: 0.1 -> 0.15"]


def test_trusted_set_change_is_reported(repo):
    _record(repo)
    repo[1].fp = "fp-2"
    assert _check(repo) == ["reference fingerprint changed"]


def test_record_refuses_existing_record_and_writes_nothing(repo):
    repo[2].write_text("old", encoding="utf-8")
    with pytest.raises(FreezeError):
        _record(repo)
    assert repo[2].read_text(encoding="utf-8") == "old"


def test_record_refuses_unset_judge_model_and_writes_nothing(repo, monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL")
    with pytest.raises(FreezeError):
        _record(repo)
    assert not repo[2].exists()


def test_record_stores_date_and_commit(repo):
    out = _record(repo)
    saved = json.loads(repo[2].read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo[0], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert saved["frozen"] == "2026-01-01" and saved["commit"] == head and saved == out


def test_check_without_record_reports_missing(repo):
    assert _check(repo) == [f"no freeze record at {repo[2]}"]


def test_require_frozen_raises_with_every_difference(repo, monkeypatch):
    _record(repo)
    repo[1].fp = "fp-2"
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-y")
    with pytest.raises(FreezeError) as e:
        require_frozen(repo[2], repo[0] / "data", root=repo[0])
    assert "reference fingerprint changed" in str(e.value)
    assert "judge_model: judge-x -> judge-y" in str(e.value)


def test_diff_of_equal_states_is_empty(repo):
    state = gate_state(repo[0] / "data", root=repo[0])
    assert diff(state, state) == []


def test_cli_check_exit_codes(repo, monkeypatch, capsys):
    root, _, path = repo
    monkeypatch.setattr(freeze_gate, "FROZEN_GATE", path)
    monkeypatch.chdir(root)
    _record(repo)
    assert freeze_gate.main(["check"]) == 0
    assert "freeze check: clean" in capsys.readouterr().out
    (root / "gate" / "policy.py").write_text("A = 9\n", encoding="utf-8")
    assert freeze_gate.main(["check"]) == 1
