#!/usr/bin/env bash
# release.sh — Build, test, and publish Triton + uTLX wheels.
#
# Usage:
#   ./release.sh <triton-ext-commit>               # build + test
#   ./release.sh <triton-ext-commit> --publish      # build + test + publish

set -euo pipefail

TRITON_EXT_COMMIT="${1:?Usage: $0 <triton-ext-commit> [--publish]}"
PUBLISH=0
[ "${2:-}" = "--publish" ] && PUBLISH=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/env.sh"

# ── resolve deps ─────────────────────────────────────────────────────────────

resolve_deps "$TRITON_EXT_COMMIT"

DIST_DIR="$SCRIPT_DIR/dist"
rm -rf "$DIST_DIR"

# ── build ────────────────────────────────────────────────────────────────────

log "Step 1/4: Building Triton ($TRITON_SHORT)..."
"$SCRIPT_DIR/triton/build_triton_wheel.sh" "$TRITON_PIN"

log "Step 2/4: Building uTLX ($EXT_SHORT)..."
"$SCRIPT_DIR/utlx/build_utlx_wheel.sh" "$TRITON_EXT_COMMIT"

TRITON_WHL=$(ls "$DIST_DIR"/triton-*.whl | head -1)
UTLX_WHL=$(ls "$DIST_DIR"/utlx-*.whl | head -1)

# ── test ─────────────────────────────────────────────────────────────────────

log "Step 3/4: Running utlx core tests..."
setup_venv
uv pip install pytest 2>&1 | tail -1

"$PYTHON" "$SCRIPT_DIR/test_runner.py"

# ── publish ──────────────────────────────────────────────────────────────────

if [ "$PUBLISH" = "1" ]; then
    log "Step 4/4: Publishing..."
    publish_wheel "$TRITON_WHL" "$(release_tag_from_wheel "$TRITON_WHL")"
    publish_wheel "$UTLX_WHL" "$(release_tag_from_wheel "$UTLX_WHL")"
else
    log "Step 4/4: Skipped (pass --publish to upload)"
    echo ""
    echo "  $TRITON_WHL"
    echo "  $UTLX_WHL"
fi
