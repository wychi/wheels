#!/usr/bin/env bash
# Install the trimul pre-commit hook into .git/hooks/.
# Idempotent: safe to re-run.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"
HOOK_SRC="$REPO_ROOT/gpumode/bioml/trimul/scripts/pre-commit"
HOOK_DST="$HOOK_DIR/pre-commit"

if [ ! -d "$HOOK_DIR" ]; then
    echo "Not a git repo (no .git/hooks): $REPO_ROOT" >&2
    exit 1
fi

if [ -e "$HOOK_DST" ] && [ "$(readlink -f "$HOOK_DST" 2>/dev/null)" != "$(readlink -f "$HOOK_SRC")" ]; then
    echo "$HOOK_DST already exists and points elsewhere. Move it aside first." >&2
    exit 1
fi

ln -sf "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_SRC"
echo "Installed $HOOK_DST -> $HOOK_SRC"
