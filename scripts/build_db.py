"""Build (or rebuild) the applicant DB without starting the API.

Usage: uv run python scripts/build_db.py [--no-reset]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DB_PATH  # noqa: E402
from app.ingest import build_database  # noqa: E402

if __name__ == "__main__":
    result = build_database(reset="--no-reset" not in sys.argv)
    print(f"inserted {result['inserted']} candidates ({result['total']} total) into {DB_PATH}")
