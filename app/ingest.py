"""Build the SQLite vector store from data/resumes.json."""

import json

from app import db, embeddings
from app.config import RESUMES_PATH


def build_database(reset: bool = True) -> dict[str, int]:
    if not RESUMES_PATH.exists():
        raise FileNotFoundError(f"{RESUMES_PATH} not found; it ships with the repo")
    rows = json.loads(RESUMES_PATH.read_text())
    db.init_db(reset=reset)
    db.insert_candidates(rows, embeddings.embed_documents([r["resume"] for r in rows]))
    return {"inserted": len(rows), "total": db.count_candidates()}
