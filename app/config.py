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

EXPERIENCE_COLLECTION = "experiences"
EXPERIENCES_PATH = os.getenv("EXPERIENCES_PATH", "data/experiences.jsonl")
EXPERIENCE_K = 3      # past experiences injected per run

# UI wiring, not gate knobs: the freeze does not record them. Only the exact
# string "1" is on; a truthiness test would read "0" or "false" as on.
UI_READ_GATE = os.getenv("UI_READ_GATE") == "1"
UI_EXPERIENCE_STORE = os.getenv("UI_EXPERIENCE_STORE") == "1"

DENSE_N = 20      # dense candidates pulled from Chroma
SPARSE_N = 20     # BM25 candidates
FUSE_M = 10       # kept after RRF, sent to reranker
TOP_K = 5         # final docs after rerank

# Minimum seconds between LLM requests. 0 = no pacing (keeps the UI responsive);
# eval runs raise it to stay under the free tier's per-minute request cap.
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "0"))

# --- Phase 2A: decomposition ---
MAX_SUBQUERIES = 3

# --- Phase 3: S2 policy judge ---
# Defaults to LLM_MODEL so the judge needs no extra config out of the box;
# override separately if the judge should run a different (e.g. cheaper) model.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
JUDGE_BATCH_MAX = 5   # chunking into batches of this size is ReadGate's job, not the signal's


# --- Phase 4: S3 corroboration ---
S3_MAX_CLAIMS = 5   # per item; a cap keeps one long chunk from dominating the call and the reference lookups
S3_REF_K = 3        # reference passages retrieved per claim
