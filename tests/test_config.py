from app import config

def test_config_constants_present():
    assert config.LLM_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.LLM_MODEL == "gemini-flash-lite-latest"
    assert config.LLM_API_KEY_ENV == "GEMINI_API_KEY"
    assert config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.TOP_K == 5
    assert config.MAX_RETRIES == 1
    assert config.DENSE_N >= config.FUSE_M >= config.TOP_K
