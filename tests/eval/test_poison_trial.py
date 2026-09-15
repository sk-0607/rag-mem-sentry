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
