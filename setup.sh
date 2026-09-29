#!/usr/bin/env bash
# One-shot setup: install deps, build the applicant DB, start the API.
#
#   ./setup.sh                 # build DB (if missing) and run on :8000
#   PORT=8002 ./setup.sh       # different port
#   ./setup.sh --rebuild       # force a DB rebuild first
#   ./setup.sh --no-run        # set up only, don't start the server
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
REBUILD=false
RUN=true
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=true ;;
    --no-run) RUN=false ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Accept the key from the shell, or from a local .env (python-dotenv loads it for the app).
if [ -z "${GEMINI_API_KEY:-}" ] && ! { [ -f .env ] && grep -Eq '^GEMINI_API_KEY=.+' .env; }; then
  echo "GEMINI_API_KEY is not set — run: export GEMINI_API_KEY=<your key>" >&2
  exit 1
fi

echo "==> Installing dependencies"
uv sync --quiet

if [ ! -f data/resumes.json ]; then
  echo "data/resumes.json is missing; it ships with the repo." >&2
  exit 1
fi

if $REBUILD || [ ! -f data/applicants.db ]; then
  echo "==> Building applicant DB (embedding resumes with Gemini)"
  uv run python scripts/build_db.py
else
  echo "==> data/applicants.db already exists (use --rebuild to regenerate)"
fi

if $RUN; then
  echo "==> Starting API on http://127.0.0.1:${PORT}  (docs at /docs)"
  exec uv run uvicorn main:app --host 127.0.0.1 --port "$PORT"
fi
