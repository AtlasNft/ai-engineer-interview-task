from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESUMES_PATH = DATA_DIR / "resumes.json"
DB_PATH = DATA_DIR / "applicants.db"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
GEMINI_MODEL = "gemini-2.5-flash"

load_dotenv(ROOT_DIR / ".env")
