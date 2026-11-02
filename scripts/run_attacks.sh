#!/usr/bin/env sh
set -eu

PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

for CONFIG_NAME in proposed_sign_flip proposed_alie proposed_slow_drift
do
  python "$PACKAGE_ROOT/main.py" \
    --config "$PACKAGE_ROOT/configs/${CONFIG_NAME}.json" \
    "$@"
done
