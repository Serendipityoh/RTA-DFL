#!/usr/bin/env sh
set -eu

PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python "$PACKAGE_ROOT/main.py" \
  --config "$PACKAGE_ROOT/configs/proposed_clean.json" \
  "$@"
