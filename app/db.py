import json
import sqlite3
from pathlib import Path

import numpy as np

from app.config import DB_PATH, EMBEDDING_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    resume TEXT NOT NULL,
    skills TEXT NOT NULL,
    current_job_title TEXT NOT NULL,
    past_job_titles TEXT NOT NULL,
    years_experience INTEGER NOT NULL,
    location_city TEXT NOT NULL,
    location_region TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""

_db_path: Path = DB_PATH


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def connect() -> sqlite3.Connection:
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False) -> None:
    with connect() as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS candidates")
        conn.executescript(SCHEMA)


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def insert_candidates(rows: list[dict], embeddings: np.ndarray) -> list[int]:
    ids: list[int] = []
    with connect() as conn:
        for row, vec in zip(rows, embeddings, strict=True):
            cur = conn.execute(
                """INSERT INTO candidates
                   (name, resume, skills, current_job_title, past_job_titles,
                    years_experience, location_city, location_region, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["name"],
                    row["resume"],
                    json.dumps(row["skills"]),
                    row["current_job_title"],
                    json.dumps(row["past_job_titles"]),
                    int(row["years_experience"]),
                    row["location_city"],
                    row["location_region"],
                    to_blob(vec),
                ),
            )
            ids.append(cur.lastrowid)
    return ids


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["skills"] = json.loads(d["skills"])
    d["past_job_titles"] = json.loads(d["past_job_titles"])
    d.pop("embedding", None)
    return d


def get_candidate(candidate_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    return row_to_dict(row) if row else None


def count_candidates() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]


def _like_pattern(value: str) -> str:
    """Case-insensitive 'includes' pattern; escapes LIKE wildcards in user input."""
    escaped = value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def query_candidates(
    *,
    skills: list[str] | None = None,
    current_job_title: str | None = None,
    past_job_titles: list[str] | None = None,
    min_years_experience: int | None = None,
    max_years_experience: int | None = None,
    location_city: str | None = None,
    location_region: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Apply structured filters in SQL and return (rows, embedding matrix).

    Text filters are case-insensitive substring matches ("captain" matches "Senior Captain").
    Skills must all be present (exact, case-insensitive). Past titles: each value must be
    contained in at least one past title.
    """
    clauses: list[str] = []
    params: list = []

    for column, value in (
        ("current_job_title", current_job_title),
        ("location_city", location_city),
        ("location_region", location_region),
    ):
        if value:
            clauses.append(f"{column} LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(value))

    if min_years_experience is not None:
        clauses.append("years_experience >= ?")
        params.append(min_years_experience)
    if max_years_experience is not None:
        clauses.append("years_experience <= ?")
        params.append(max_years_experience)

    # List columns are stored as JSON; require every requested value to be present.
    for v in skills or []:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(candidates.skills) WHERE LOWER(json_each.value) = LOWER(?))"
        )
        params.append(v.strip())
    for v in past_job_titles or []:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(candidates.past_job_titles) WHERE json_each.value LIKE ? ESCAPE '\\')"
        )
        params.append(_like_pattern(v))

    sql = "SELECT * FROM candidates"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = np.stack([from_blob(r["embedding"]) for r in rows])
    return [row_to_dict(r) for r in rows], matrix
