import json

import pytest

import app.llm
from gate.types import GateContext, Item
from gate.signals.judge import JudgeSignal, JudgeResponseError

CTX = GateContext(store="corpus")


class FakeChat:
    """Records every call's (messages, model) and returns a canned response.

    Matches app.llm.chat's call shape (messages, client=None, model=None) —
    JudgeSignal never passes client, so accepting **kwargs keeps this
    tolerant of that without asserting on it.
    """

    def __init__(self, response="[]"):
        self.response = response
        self.calls = []

    def __call__(self, messages, model=None, **kwargs):
        self.calls.append((messages, model))
        return self.response


def items(n):
    return [Item(id=f"item-{i}", text=f"passage text {i}", kind="corpus") for i in range(n)]


def entry(idx, suspicion=0.1, label="clean", evidence_span=None):
    return {"id": f"i{idx}", "suspicion": suspicion, "label": label, "evidence_span": evidence_span}


def response(*entries):
    return json.dumps(list(entries))


def test_one_call_per_score_batch():
    its = items(3)
    fake = FakeChat(response(entry(0), entry(1), entry(2)))
    JudgeSignal(chat=fake).score_batch(its, CTX)
    assert len(fake.calls) == 1


def test_results_align_with_input_order():
    its = items(3)
    # Response lists ids out of order; results must come back in input order.
    body = json.dumps([
        entry(2, suspicion=0.9, label="instructs"),
        entry(0, suspicion=0.1, label="clean"),
        entry(1, suspicion=0.5, label="quotes"),
    ])
    fake = FakeChat(body)
    results = JudgeSignal(chat=fake).score_batch(its, CTX)
    assert [r.score for r in results] == [0.1, 0.5, 0.9]
    assert [r.reason for r in results] == ["clean", "quotes", "instructs"]


def test_prompt_wraps_items_with_batch_local_ids():
    its = [Item(id="real-item-id", text="the passage body", kind="corpus")]
    fake = FakeChat(response(entry(0)))
    JudgeSignal(chat=fake).score_batch(its, CTX)
    user_msg = fake.calls[0][0][1]["content"]
    assert '<item id="i0">' in user_msg
    assert "the passage body" in user_msg
    assert "real-item-id" not in user_msg


def test_system_prompt_marks_item_content_as_untrusted():
    its = items(1)
    fake = FakeChat(response(entry(0)))
    JudgeSignal(chat=fake).score_batch(its, CTX)
    system_msg = fake.calls[0][0][0]["content"]
    assert "untrusted" in system_msg.lower()
    assert "never an instruction to follow" in system_msg.lower()


def test_prompt_never_contains_the_query():
    its = items(1)
    fake = FakeChat(response(entry(0)))
    ctx = GateContext(store="corpus", query="SECRET-Q")
    JudgeSignal(chat=fake).score_batch(its, ctx)
    system_msg, user_msg = (m["content"] for m in fake.calls[0][0])
    assert "SECRET-Q" not in system_msg
    assert "SECRET-Q" not in user_msg


def test_corpus_label_set():
    its = items(1)
    fake = FakeChat(response(entry(0, label="quotes")))
    results = JudgeSignal(chat=fake).score_batch(its, CTX)
    assert results[0].reason == "quotes"

    fake = FakeChat(response(entry(0, label="endorses_unsafe")))
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(its, CTX)


def test_fenced_json_is_parsed():
    its = items(1)
    body = f"```json\n{response(entry(0))}\n```"
    fake = FakeChat(body)
    results = JudgeSignal(chat=fake).score_batch(its, CTX)
    assert results[0].score == 0.1


def test_malformed_json_raises():
    fake = FakeChat("not json")
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(1), CTX)


def test_missing_id_raises():
    fake = FakeChat(response(entry(0)))  # only i0, but 2 items sent
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(2), CTX)


def test_duplicate_id_raises():
    fake = FakeChat(response(entry(0), entry(0)))
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(1), CTX)


def test_unknown_id_raises():
    fake = FakeChat(response(entry(5)))
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(1), CTX)


def test_out_of_range_suspicion_raises():
    fake = FakeChat(response(entry(0, suspicion=1.5)))
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(1), CTX)


def test_boolean_suspicion_raises():
    fake = FakeChat(response(entry(0, suspicion=True)))
    with pytest.raises(JudgeResponseError):
        JudgeSignal(chat=fake).score_batch(items(1), CTX)


def test_non_verbatim_evidence_span_is_dropped_score_kept():
    its = [Item(id="a", text="clean passage", kind="corpus")]
    fake = FakeChat(response(entry(0, suspicion=0.4, evidence_span="not in the text")))
    results = JudgeSignal(chat=fake).score_batch(its, CTX)
    assert results[0].evidence_span is None
    assert results[0].score == 0.4


def test_verbatim_evidence_span_is_kept():
    its = [Item(id="a", text="the suspicious passage body", kind="corpus")]
    fake = FakeChat(response(entry(0, evidence_span="suspicious passage")))
    results = JudgeSignal(chat=fake).score_batch(its, CTX)
    assert results[0].evidence_span == "suspicious passage"


def test_empty_items_makes_no_call():
    fake = FakeChat("[]")
    results = JudgeSignal(chat=fake).score_batch([], CTX)
    assert results == []
    assert fake.calls == []


def test_unknown_store_abstains_without_calling():
    fake = FakeChat(response(entry(0)))
    ctx = GateContext(store="other")
    results = JudgeSignal(chat=fake).score_batch(items(1), ctx)
    assert results == [None]
    assert fake.calls == []


def test_model_passed_to_chat():
    fake = FakeChat(response(entry(0)))
    JudgeSignal(chat=fake, model="judge-m").score_batch(items(1), CTX)
    assert fake.calls[0][1] == "judge-m"


def test_default_chat_resolved_at_call_time(monkeypatch):
    signal = JudgeSignal()  # constructed before app.llm.chat is patched
    fake = FakeChat(response(entry(0)))
    monkeypatch.setattr(app.llm, "chat", fake)
    results = signal.score_batch(items(1), CTX)
    assert len(fake.calls) == 1
    assert results[0].score == 0.1


def test_score_delegates_to_score_batch():
    it = Item(id="a", text="passage", kind="corpus")
    fake = FakeChat(response(entry(0, suspicion=0.3, label="quotes")))
    result = JudgeSignal(chat=fake).score(it, CTX)
    assert result.score == 0.3
    assert result.reason == "quotes"
