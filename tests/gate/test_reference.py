import json

from gate.reference import ReferenceIndex, default_reference


def _c(id_, text, trusted=True):
    c = {"id": id_, "text": text}
    if trusted is not None:
        c["trusted"] = trusted
    return c


def _chunks():
    return [
        _c("src-a::0", "alpha zebra grazing on the savanna"),
        _c("src-b::0", "bravo quartz crystal forms slowly"),
        _c("src-c::0", "charlie nimbus cloud drifts overhead"),
        _c("src-d::0", "delta falcon dives toward prey"),
        _c("src-e::0", "echo glacier melts every summer"),
        _c("src-f::0", "foxtrot onyx hidden secret", trusted=False),
    ]


def test_only_boolean_true_chunks_are_indexed():
    idx = ReferenceIndex([
        _c("a", "one", True), _c("b", "two", False),
        _c("c", "three", None), _c("d", "four", "true"),
    ])
    assert len(idx) == 1


def test_untrusted_chunk_never_returned():
    assert ReferenceIndex(_chunks()).search("onyx", 5) == []


def test_search_returns_best_match_first():
    # Input order puts the weaker match first so order cannot pass by accident.
    idx = ReferenceIndex([
        _c("src-a::0", "zebra plains walking tall"),
        _c("src-b::0", "zebra savanna grazing quietly"),
        _c("src-c::0", "quartz crystal forms slowly"),
        _c("src-d::0", "glacier melts every summer"),
        _c("src-e::0", "falcon dives toward prey"),
    ])
    assert [c["id"] for c in idx.search("zebra savanna grazing", 5)] == ["src-b::0", "src-a::0"]


def test_search_breaks_score_ties_by_chunk_id():
    idx = ReferenceIndex([
        _c("src-b::0", "zebra grazing"),
        _c("src-a::0", "zebra grazing"),
        _c("src-c::0", "quartz crystal forms slowly"),
        _c("src-d::0", "glacier melts every summer"),
    ])
    assert [c["id"] for c in idx.search("zebra", 5)] == ["src-a::0", "src-b::0"]


def test_search_respects_k():
    idx = ReferenceIndex(_chunks())
    assert len(idx.search("the zebra falcon glacier quartz", 2)) == 2


def test_search_without_shared_token_returns_empty():
    assert ReferenceIndex(_chunks()).search("unrelated words", 5) == []


def test_empty_index_has_length_zero_and_empty_search():
    idx = ReferenceIndex([])
    assert len(idx) == 0 and idx.search("anything", 3) == []


def test_nonpositive_k_returns_empty():
    idx = ReferenceIndex(_chunks())
    assert idx.search("zebra", 0) == [] and idx.search("zebra", -1) == []


def test_contains_text_is_exact():
    idx = ReferenceIndex(_chunks())
    assert idx.contains_text("alpha zebra grazing on the savanna")
    assert not idx.contains_text("alpha zebra grazing on the savanna ")
    assert not idx.contains_text("foxtrot onyx hidden secret")


def test_fingerprint_changes_when_a_chunk_becomes_trusted():
    a = ReferenceIndex(_chunks())
    b = ReferenceIndex(_chunks()[:5] + [_c("src-f::0", "foxtrot onyx hidden secret")])
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_changes_when_trusted_text_changes():
    changed = _chunks()
    changed[0] = _c("src-a::0", "alpha zebra edited")
    assert ReferenceIndex(_chunks()).fingerprint() != ReferenceIndex(changed).fingerprint()


def test_fingerprint_ignores_untrusted_chunks():
    changed = _chunks()
    changed[5] = _c("src-f::0", "different untrusted text", trusted=False)
    assert ReferenceIndex(_chunks()).fingerprint() == ReferenceIndex(changed).fingerprint()


def test_fingerprint_ignores_input_order():
    assert (ReferenceIndex(_chunks()).fingerprint()
            == ReferenceIndex(list(reversed(_chunks()))).fingerprint())


def test_default_reference_reads_chunks_jsonl(tmp_path):
    lines = [_c("src-a::0", "alpha zebra", True), _c("src-b::0", "bravo quartz", False)]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in lines) + "\n", encoding="utf-8")
    assert len(default_reference(tmp_path)) == 1


def test_default_reference_missing_file_is_empty(tmp_path):
    assert len(default_reference(tmp_path)) == 0
