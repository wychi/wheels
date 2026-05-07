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
    if TRITON_URL=$(publish_wheel "$TRITON_WHL" "$(release_tag_from_wheel "$TRITON_WHL")" | xargs) && \
       UTLX_URL=$(publish_wheel "$UTLX_WHL" "$(release_tag_from_wheel "$UTLX_WHL")" | xargs); then
        
        # Record release to releases.json (only if both publishes succeeded)
        RELEASES_FILE="$SCRIPT_DIR/releases.json"
        TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        
        # Create JSON entry and prepend to file
        if [ -f "$RELEASES_FILE" ]; then
            # File exists - prepend new entry
            TMP_FILE=$(mktemp)
            cat > "$TMP_FILE" <<EOF
[
  {
    "time": "$TIMESTAMP",
    "utlx": "$UTLX_URL",
    "triton": "$TRITON_URL"
  },
EOF
            # Append existing entries (skip first line which is "[")
            tail -n +2 "$RELEASES_FILE" >> "$TMP_FILE"
            mv "$TMP_FILE" "$RELEASES_FILE"
        else
            # File doesn't exist - create it
            cat > "$RELEASES_FILE" <<EOF
[
  {
    "time": "$TIMESTAMP",
    "utlx": "$UTLX_URL",
    "triton": "$TRITON_URL"
  }
]
EOF
        fi
        
        log "Release recorded to $RELEASES_FILE"
    else
        err "Publishing failed. Release not recorded to $RELEASES_FILE."
    fi
else
    log "Step 4/4: Skipped (pass --publish to upload)"
    echo ""
    echo "  $TRITON_WHL"
    echo "  $UTLX_WHL"
fi
