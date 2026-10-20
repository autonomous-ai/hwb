#!/usr/bin/env bash
# Launcher for running hwb from a source checkout or an extracted bundle, no install needed.
# Picks the newest system python3 >= 3.10; falls back to uv-managed Python if uv is present.
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MIN_MINOR=10

find_python() {
  local py
  for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1 &&
      "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, ${MIN_MINOR}) else 1)" 2>/dev/null; then
      command -v "$py"
      return 0
    fi
  done
  return 1
}

if PY="$(find_python)"; then
  # -s: ignore user site-packages so a stray ~/.local install can't shadow stdlib modules.
  PYTHONPATH="${HERE}/src${PYTHONPATH:+:${PYTHONPATH}}" exec "$PY" -s -m hwb "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python ">=3.${MIN_MINOR}" --with-editable "${HERE}" -- python -m hwb "$@"
else
  echo "hwb: need python3 >= 3.${MIN_MINOR} (or uv). Install one of:" >&2
  echo "  apt-get install -y python3        # Ubuntu / Debian" >&2
  echo "  pacman -S --needed python         # Arch / Omarchy" >&2
  exit 2
fi
