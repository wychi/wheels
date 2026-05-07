#!/usr/bin/env bash
# gh_release.sh — Upload built wheels to GitHub Releases and record in
# releases.json.
#
# Usage:
#   ./gh_release.sh                    # publish dist/triton-*.whl + dist/utlx-*.whl
#   ./gh_release.sh <triton.whl> <utlx.whl>
#
# Requires GH_TOKEN in env (or `gh auth login` already done).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/env.sh"

if [ -z "${GH_TOKEN:-}" ] && ! gh auth status >/dev/null 2>&1; then
    err "GH_TOKEN not set and 'gh auth status' failed. Run 'gh auth login' or export GH_TOKEN."
fi

# ── resolve wheels ───────────────────────────────────────────────────────────

DIST_DIR="$SCRIPT_DIR/dist"

if [ "$#" -ge 2 ]; then
    TRITON_WHL="$1"
    UTLX_WHL="$2"
else
    TRITON_WHL=$(ls "$DIST_DIR"/triton-*.whl 2>/dev/null | head -1)
    UTLX_WHL=$(ls "$DIST_DIR"/utlx-*.whl 2>/dev/null | head -1)
fi

[ -f "$TRITON_WHL" ] || err "Triton wheel not found: ${TRITON_WHL:-(none)}"
[ -f "$UTLX_WHL" ]   || err "uTLX wheel not found: ${UTLX_WHL:-(none)}"

log "Publishing:"
log "  $TRITON_WHL"
log "  $UTLX_WHL"

# ── publish ──────────────────────────────────────────────────────────────────

if ! TRITON_URL=$(publish_wheel "$TRITON_WHL" "$(release_tag_from_wheel "$TRITON_WHL")"); then
    err "Failed to publish Triton wheel."
fi
if ! UTLX_URL=$(publish_wheel "$UTLX_WHL" "$(release_tag_from_wheel "$UTLX_WHL")"); then
    err "Failed to publish uTLX wheel."
fi

# ── record ───────────────────────────────────────────────────────────────────
# Prepend a new entry to releases.json. make_submission.py reads index [0]
# (the most recent entry) when resolving wheel URLs.

RELEASES_FILE="$SCRIPT_DIR/releases.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ -f "$RELEASES_FILE" ]; then
    TMP_FILE=$(mktemp)
    cat > "$TMP_FILE" <<EOF
[
  {
    "time": "$TIMESTAMP",
    "utlx": "$UTLX_URL",
    "triton": "$TRITON_URL"
  },
EOF
    tail -n +2 "$RELEASES_FILE" >> "$TMP_FILE"
    mv "$TMP_FILE" "$RELEASES_FILE"
else
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
echo ""
echo "  triton: $TRITON_URL"
echo "  utlx:   $UTLX_URL"
