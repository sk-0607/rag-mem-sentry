from gate.types import SignalResult
from gate.cache import VerdictCache, content_hash


def test_hash_is_stable_for_identical_text():
    assert content_hash("hello world") == content_hash("hello world")


def test_hash_differs_for_different_text():
    assert content_hash("hello") != content_hash("hello ")


def test_zero_width_character_changes_the_hash():
    # A smuggled payload must not reuse the clean text's cached verdict.
    assert content_hash("hello world") != content_hash("hello​world")


def test_hash_is_sha256_shaped():
    digest = content_hash("anything")
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_miss_returns_none(tmp_path):
    cache = VerdictCache(tmp_path / "c.jsonl")
    assert cache.get("nope") is None


def test_put_then_get_roundtrips(tmp_path):
    cache = VerdictCache(tmp_path / "c.jsonl")
    results = [SignalResult(signal="s1", score=0.4, reason="why", evidence_span="span")]
    cache.put("k", results)
    got = cache.get("k")
    assert got == results


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.jsonl"
    first = VerdictCache(path)
    first.put("k", [SignalResult(signal="s1", score=0.4, reason="why")])
    first.close()

    second = VerdictCache(path)
    assert second.get("k")[0].score == 0.4


def test_later_write_for_same_key_wins(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = VerdictCache(path)
    cache.put("k", [SignalResult(signal="s1", score=0.1, reason="old")])
    cache.put("k", [SignalResult(signal="s1", score=0.9, reason="new")])
    cache.close()
    assert VerdictCache(path).get("k")[0].reason == "new"


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"key": "k", "results": [{"signal": "s1", "score": 0.4, "reason": "ok", "evidence_span": null}]}\nNOT JSON\n', encoding="utf-8")
    cache = VerdictCache(path)
    assert cache.get("k")[0].reason == "ok"
