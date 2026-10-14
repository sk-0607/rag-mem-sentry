import hashlib

from eval.attack_corpus import to_gate_item
from eval.probe_server import parse_probe
from eval.probe_summary import read_log, summarise, verdict_flips


def row(session=1, cls=1, status=200, admitted=True, h="h1", calls=1, counted=True,
        cached=False, ts="2026-10-15T00:00:00"):
    return {"ts": ts, "session": session, "class": cls, "store": "corpus", "item_hash": h,
            "fields": {"text": "x"}, "status": status, "cached": cached, "counted": counted,
            "calls": calls, "admitted": admitted if status == 200 else None,
            "action": None}


def test_counts_per_session():
    rows = [row(), row(h="h2", counted=False, cached=True, calls=0), row(session=2, h="h3")]
    s = summarise(rows)["sessions"]
    assert s["1"]["counted"] == 1 and s["1"]["cached"] == 1
    assert s["2"]["counted"] == 1 and s["2"]["cached"] == 0


def test_rates_per_class():
    rows = [row(cls=1, admitted=True, h="a"), row(cls=1, admitted=False, h="b"),
            row(cls=3, admitted=False, h="c")]
    classes = summarise(rows)["sessions"]["1"]["classes"]
    assert classes["1"] == {"probed": 2, "admitted": 1, "rate": 0.5}
    assert classes["3"] == {"probed": 1, "admitted": 0, "rate": 0.0}


def test_rate_null_when_nothing_probed():
    classes = summarise([row(cls=2, status=429, calls=0, counted=False)])["sessions"]["1"]["classes"]
    assert classes["2"] == {"probed": 0, "admitted": 0, "rate": None}


def test_429_and_503_counted_separately_not_as_probes():
    rows = [row(status=429, calls=0, counted=False), row(status=503, calls=2, counted=False),
            row(status=503, calls=1, counted=False)]
    s = summarise(rows)["sessions"]["1"]
    assert s["rate_limited"] == 1 and s["unavailable"] == 2
    assert s["classes"]["1"]["probed"] == 0


def test_calls_summed_over_every_row():
    rows = [row(calls=2), row(status=503, calls=3, counted=False), row(status=429, calls=0, counted=False)]
    assert summarise(rows)["sessions"]["1"]["calls"] == 5


def test_ever_admitted_counts_distinct_items():
    rows = [row(h="a", admitted=True), row(h="a", admitted=True, session=2),
            row(h="b", admitted=False), row(h="c", cls=3, admitted=True)]
    r = summarise(rows)["round"]
    assert r["distinct_items"] == 3
    assert r["ever_admitted"] == {"1": 1, "3": 1}


def test_reads_bom_and_blank_lines(tmp_path):
    p = tmp_path / "probes.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n\n{"a": 2}\n')
    assert read_log(p) == [{"a": 1}, {"a": 2}]


def entry(i, text):
    return {"id": f"e{i}", "class": 1, "group": "g", "text": text}


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_verdict_flips_counts_both_directions():
    entries = [entry(1, "one"), entry(2, "two"), entry(3, "three")]
    probes = [row(h=h("one"), admitted=True), row(h=h("two"), admitted=False),
              row(h=h("three"), admitted=True)]
    screen = [{"id": "e1", "action": "quarantine"}, {"id": "e2", "action": "admit"},
              {"id": "e3", "action": "admit"}]
    assert verdict_flips(probes, entries, screen) == {
        "final_items": 3, "probed_in_final": 3, "admitted_then_excluded": 1,
        "excluded_then_admitted": 1}


def test_verdict_flips_uses_the_latest_probe_of_an_item():
    probes = [row(h=h("one"), admitted=False, ts="2026-10-15T02:00:00"),
              row(h=h("one"), admitted=True, ts="2026-10-15T01:00:00")]
    out = verdict_flips(probes, [entry(1, "one")], [{"id": "e1", "action": "block"}])
    assert out["admitted_then_excluded"] == 0 and out["excluded_then_admitted"] == 0


def test_unprobed_final_items_not_counted_as_flips():
    probes = [row(h=h("one"), status=429, calls=0, counted=False)]
    out = verdict_flips(probes, [entry(1, "one"), entry(2, "two")],
                        [{"id": "e1", "action": "admit"}, {"id": "e2", "action": "block"}])
    assert out == {"final_items": 2, "probed_in_final": 0, "admitted_then_excluded": 0,
                   "excluded_then_admitted": 0}


def test_hash_matches_probe_server_item_hash():
    body = {"class": 3, "store": "experience", "task": "t", "approach": "a", "outcome": "o"}
    probe_item, _ = to_gate_item(parse_probe(dict(body)))
    final = {"id": "e9", "class": 3, "group": "g", "task": "t", "approach": "a", "outcome": "o"}
    screen = [{"id": "e9", "action": "admit"}]
    probes = [row(cls=3, h=h(probe_item.text), admitted=False)]
    assert verdict_flips(probes, [final], screen)["excluded_then_admitted"] == 1
