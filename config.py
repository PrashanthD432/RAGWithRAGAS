import os

from dotenv import load_dotenv


# RAGAS imports GitPython for optional experiment tracking. Do not fail when
# git.exe is unavailable because this project does not use that feature.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in your .env file.")

GENERATION_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
TOP_K = 5
