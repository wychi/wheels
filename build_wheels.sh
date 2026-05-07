#!/usr/bin/env bash
# build_wheels.sh — Build and test Triton + uTLX wheels.
#
# Usage:
#   ./build_wheels.sh <triton-ext-commit>
#
# To upload the resulting wheels to GitHub Releases, run gh_release.sh after.

set -euo pipefail

TRITON_EXT_COMMIT="${1:?Usage: $0 <triton-ext-commit>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/env.sh"

# ── resolve deps ─────────────────────────────────────────────────────────────

resolve_deps "$TRITON_EXT_COMMIT"

DIST_DIR="$SCRIPT_DIR/dist"
rm -rf "$DIST_DIR"

# ── build ────────────────────────────────────────────────────────────────────

log "Step 1/3: Building Triton ($TRITON_SHORT)..."
"$SCRIPT_DIR/triton/build_triton_wheel.sh" "$TRITON_PIN"

log "Step 2/3: Building uTLX ($EXT_SHORT)..."
"$SCRIPT_DIR/utlx/build_utlx_wheel.sh" "$TRITON_EXT_COMMIT"

TRITON_WHL=$(ls "$DIST_DIR"/triton-*.whl | head -1)
UTLX_WHL=$(ls "$DIST_DIR"/utlx-*.whl | head -1)

# ── test ─────────────────────────────────────────────────────────────────────

log "Step 3/3: Running utlx core tests..."
setup_venv
uv pip install pytest 2>&1 | tail -1

"$PYTHON" "$SCRIPT_DIR/test_runner.py"

# ── done ─────────────────────────────────────────────────────────────────────

echo ""
echo "Wheels built:"
echo "  $TRITON_WHL"
echo "  $UTLX_WHL"
echo ""
echo "To publish to GitHub Releases, run:"
echo "  ./gh_release.sh"
