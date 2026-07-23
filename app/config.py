import os

from dotenv import load_dotenv

# Load .env before reading any vars so LLM_MODEL / LLM_BASE_URL overrides apply
# regardless of import order.
load_dotenv()

# LLM provider is any OpenAI-compatible endpoint. Default: Google AI Studio
# (Gemini) free tier. Swap LLM_BASE_URL / LLM_MODEL + the key env to change
# providers without touching call sites.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
LLM_API_KEY_ENV = "GEMINI_API_KEY"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
COLLECTION = "papers"

DENSE_N = 20      # dense candidates pulled from Chroma
SPARSE_N = 20     # BM25 candidates
FUSE_M = 10       # kept after RRF, sent to reranker
TOP_K = 5         # final docs after rerank
MAX_RETRIES = 1   # CRAG rewrite+retry budget
