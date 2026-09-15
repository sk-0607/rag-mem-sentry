import types

import pytest


class StubRetriever:
    """Drives the real graph without a Chroma index: one doc, always."""
    def retrieve(self, query, k=5):
        return [{"title": "T", "chunk_index": 0, "text": "ctx", "score": 1.0}]


@pytest.fixture
def stub_retriever():
    return StubRetriever()


@pytest.fixture
def stub_llm():
    """Atomic-question llm_module: one subquery, passthrough synthesize. Drives the real graph end to end with zero
    network calls."""
    m = types.SimpleNamespace()
    m.decompose = lambda q, client=None: [q]
    m.answer = lambda q, docs, experiences="", client=None: {"answer": "stub answer",
                                              "citations": ["[T, chunk#0]"]}
    m.synthesize = lambda q, sr, client=None: {"answer": sr[0]["answer"],
                                                "citations": sr[0]["citations"]}
    return m
