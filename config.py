"""Central configuration for models, storage, retrieval, and credentials."""

import os

from dotenv import load_dotenv


# RAGAS imports GitPython for optional experiment tracking. Do not fail when
# git.exe is unavailable because this project does not use that feature.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Load local secrets without hardcoding credentials in source control.
load_dotenv()

# Avoid repeated Hugging Face network checks after the two local models have
# been downloaded. Set HF_LOCAL_FILES_ONLY=false only for a first-time setup.
HF_LOCAL_FILES_ONLY = os.getenv(
    "HF_LOCAL_FILES_ONLY",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
if HF_LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY (or Gemini_API_Key) is missing in your .env file."
    )

# Gemini writes the final answer and also judges optional RAGAS metrics.
GENERATION_MODEL = "gemini-3.8-flash"

# A bi-encoder is fast enough to embed every chunk for first-stage retrieval.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A cross-encoder is slower but more accurate because it reads each
# question-document pair jointly. It is used only on the candidate set.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Generated indexes are persisted locally so they can be inspected or reused.
FAISS_INDEX_PATH = "faiss_index"

# Every PDF below this directory is discovered recursively.
DOCUMENTS_DIR = "Documents"

# Retrieve broadly for recall, then keep a smaller grounded LLM context.
CANDIDATE_K = 15
FINAL_TOP_K = 5

# FAISS returns squared L2 distance for these normalized embeddings, so a
# smaller value is more relevant. BM25 is a relevance score, so larger is
# better. Keeping the directions explicit prevents weak dense matches from
# being accepted accidentally by a generic `score > threshold` comparison.
DENSE_MAX_DISTANCE = 1.0
SPARSE_MIN_SCORE = 1.0

# RecursiveCharacterTextSplitter uses character counts. An 800-character
# window retains useful local context, while 80 characters (within the
# requested 50-100 range) preserve continuity across adjacent chunks.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# RAGAS uses several additional judge-model calls for every question. Keep it
# disabled during normal Q&A to protect the Gemini quota. Set
# RUN_RAGAS=true in .env only when you intentionally want an evaluation run.
RUN_RAGAS = os.getenv("RUN_RAGAS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
