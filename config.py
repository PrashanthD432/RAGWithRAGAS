"""Central configuration for models, storage, retrieval, and credentials."""

import os

from dotenv import load_dotenv


# RAGAS imports GitPython for optional experiment tracking. Do not fail when
# git.exe is unavailable because this project does not use that feature.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Load local secrets without hardcoding credentials in source control.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in your .env file.")

# The generation model writes the final answer and also judges RAGAS metrics.
GENERATION_MODEL = "openai/gpt-oss-20b"

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
