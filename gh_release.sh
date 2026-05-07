#!/usr/bin/env bash
# gh_release.sh — Test, then upload built wheels to GitHub Releases and
# record the (triton, utlx) pair in releases.json.
#
# Usage:
#   ./gh_release.sh
#
# Picks `dist/triton-*.whl` and `dist/utlx-*.whl` (one each — fails if more).
#
# Steps:
#   1. Resolve the wheels in dist/.
#   2. Refuse if releases.json already records this exact (triton, utlx)
#      pair (any entry, not just the latest — re-publishing is wasteful and
#      bloats the file).
#   3. Run test_runner.py to install the wheels and exercise the core test
#      suite. Skips publish if anything fails.
#   4. Upload both wheels to GitHub Releases (creates the per-wheel tag if
#      it doesn't exist; uploads with --clobber otherwise).
#   5. Prepend a new entry to releases.json — make_submission.py reads
#      index [0] (the most recent) when resolving wheel URLs.
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

# Reuse test_runner.py's "exactly one wheel per name" rule by globbing the
# same way it does and erroring if there are zero or multiple matches.
shopt -s nullglob
TRITON_MATCHES=("$DIST_DIR"/triton-*.whl)
UTLX_MATCHES=("$DIST_DIR"/utlx-*.whl)
shopt -u nullglob

[ "${#TRITON_MATCHES[@]}" -eq 1 ] \
    || err "Expected exactly one dist/triton-*.whl, found ${#TRITON_MATCHES[@]}. Run build_wheels.sh first."
[ "${#UTLX_MATCHES[@]}" -eq 1 ] \
    || err "Expected exactly one dist/utlx-*.whl, found ${#UTLX_MATCHES[@]}. Run build_wheels.sh first."

TRITON_WHL="${TRITON_MATCHES[0]}"
UTLX_WHL="${UTLX_MATCHES[0]}"

TRITON_TAG=$(release_tag_from_wheel "$TRITON_WHL")
UTLX_TAG=$(release_tag_from_wheel "$UTLX_WHL")

# What URLs would `publish_wheel` produce? Mirrors env.sh:publish_wheel's
# echo line. We compare against releases.json to detect re-runs.
TRITON_URL_PREDICTED="https://github.com/$REPO/releases/download/$TRITON_TAG/$(basename "$TRITON_WHL")"
UTLX_URL_PREDICTED="https://github.com/$REPO/releases/download/$UTLX_TAG/$(basename "$UTLX_WHL")"

log "Wheels:"
log "  $TRITON_WHL"
log "  $UTLX_WHL"

# ── duplicate check ──────────────────────────────────────────────────────────

RELEASES_FILE="$SCRIPT_DIR/releases.json"
if [ -f "$RELEASES_FILE" ]; then
    if python3 - "$RELEASES_FILE" "$TRITON_URL_PREDICTED" "$UTLX_URL_PREDICTED" <<'PY'
import json, sys
path, triton_url, utlx_url = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    entries = json.load(f)
for e in entries:
    if e.get("triton") == triton_url and e.get("utlx") == utlx_url:
        sys.exit(0)
sys.exit(1)
PY
    then
        err "This (triton, utlx) wheel pair is already recorded in $(basename "$RELEASES_FILE"). Aborting."
    fi
fi

# ── test ─────────────────────────────────────────────────────────────────────

log "Testing wheels (test_runner.py installs from dist/)..."
setup_venv
uv pip install pytest 2>&1 | tail -1
"$PYTHON" "$SCRIPT_DIR/test_runner.py"

# ── publish ──────────────────────────────────────────────────────────────────

log "Publishing..."

if ! TRITON_URL=$(publish_wheel "$TRITON_WHL" "$TRITON_TAG"); then
    err "Failed to publish Triton wheel."
fi
if ! UTLX_URL=$(publish_wheel "$UTLX_WHL" "$UTLX_TAG"); then
    err "Failed to publish uTLX wheel."
fi

# Sanity: the URLs we recorded against in the dup check must match what
# publish_wheel returned. If this trips, our prediction is wrong and the
# dup check would falsely pass on re-run.
[ "$TRITON_URL" = "$TRITON_URL_PREDICTED" ] \
    || err "Predicted Triton URL ($TRITON_URL_PREDICTED) != published ($TRITON_URL)"
[ "$UTLX_URL" = "$UTLX_URL_PREDICTED" ] \
    || err "Predicted uTLX URL ($UTLX_URL_PREDICTED) != published ($UTLX_URL)"

# ── record ───────────────────────────────────────────────────────────────────

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
