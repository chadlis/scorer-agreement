#!/usr/bin/env bash
# Checks to run before every commit. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
for f in README.md; do
  if grep -n -E '—|–|;' "$f"; then
    echo "$f: em dash, en dash or semicolon found" >&2
    exit 1
  fi
done
echo "all checks passed"
