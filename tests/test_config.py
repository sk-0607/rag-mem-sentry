from app import config

def test_config_constants_present():
    assert config.LLM_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.LLM_MODEL == "gemini-flash-lite-latest"
    assert config.LLM_API_KEY_ENV == "GEMINI_API_KEY"
    assert config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.TOP_K == 5
    assert config.DENSE_N >= config.FUSE_M >= config.TOP_K

def test_phase2a_constants():
    assert config.MAX_SUBQUERIES == 3

def test_judge_model_defaults_to_llm_model(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    import importlib
    importlib.reload(config)
    try:
        assert config.JUDGE_MODEL == config.LLM_MODEL
    finally:
        importlib.reload(config)

def test_judge_batch_max_is_five():
    assert config.JUDGE_BATCH_MAX == 5
