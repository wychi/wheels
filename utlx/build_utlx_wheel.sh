#!/usr/bin/env bash
# build_utlx_wheel.sh — Build a uTLX wheel from a triton-ext commit.
#
# Prerequisites: LLVM and Triton must already be built at the pinned commits.
# The pin is read from triton-ext's ci/triton-hash.txt.
#
# Usage:
#   ./build_utlx_wheel.sh <triton-ext-commit>
#   ./build_utlx_wheel.sh <triton-ext-commit> --publish
#
# Output:
#   dist/utlx-*.whl

set -euo pipefail

TRITON_EXT_COMMIT="${1:?Usage: $0 <triton-ext-commit> [--publish]}"
PUBLISH=0
[ "${2:-}" = "--publish" ] && PUBLISH=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$(dirname "$SCRIPT_DIR")"

source "$WHEELS_DIR/env.sh"

# ── resolve and verify deps ──────────────────────────────────────────────────

resolve_deps "$TRITON_EXT_COMMIT"

LLVM_BUILD_DIR="$LLVM_REPO/build"

LLVM_CURRENT=$(git -C "$LLVM_REPO" rev-parse HEAD 2>/dev/null || echo "none")
[ "$LLVM_CURRENT" = "$LLVM_PIN" ] \
    || err "LLVM at $(get_short_hash "$LLVM_REPO" HEAD), need $LLVM_SHORT. Run: triton/build_triton_wheel.sh $TRITON_SHORT"
[ -f "$LLVM_BUILD_DIR/bin/mlir-tblgen" ] \
    || err "LLVM not built. Run: triton/build_triton_wheel.sh $TRITON_SHORT"

TRITON_CURRENT=$(git -C "$TRITON_REPO" rev-parse HEAD 2>/dev/null || echo "none")
[ "$TRITON_CURRENT" = "$TRITON_PIN" ] \
    || err "Triton at $(get_short_hash "$TRITON_REPO" HEAD), need $TRITON_SHORT. Run: triton/build_triton_wheel.sh $TRITON_SHORT"

setup_llvm_tools "$LLVM_BUILD_DIR"
find_libtriton
log "libtriton.so: $LIBTRITON"

# ── build libutlx.so ────────────────────────────────────────────────────────

git -C "$TRITON_EXT_REPO" checkout "$EXT_FULL"

UTLX_BUILD_DIR="$TRITON_EXT_REPO/build"

log "Configuring triton-ext..."
TRITON_SOURCE_DIR="$TRITON_REPO" \
TRITON_BUILD_DIR="$TRITON_CMAKE_DIR" \
LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
cmake -S "$TRITON_EXT_REPO" -B "$UTLX_BUILD_DIR" -G Ninja \
    -DTRITON_LIB="$LIBTRITON" \
    -DLLVM_EXTERNAL_LIT=/usr/bin/true \
    -DFILECHECK_PATH="$LLVM_BUILD_DIR/bin/FileCheck"

log "Building libutlx.so..."
cmake --build "$UTLX_BUILD_DIR"
[ -f "$UTLX_BUILD_DIR/lib/libutlx.so" ] || err "libutlx.so build failed"

# ── package wheel ────────────────────────────────────────────────────────────

UTLX_SRC_DIR="$TRITON_EXT_REPO/extensions/utlx"
UTLX_BASE_VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' "$UTLX_SRC_DIR/pyproject.toml" 2>/dev/null || echo "0.1.0")
UTLX_VERSION="${UTLX_BASE_VERSION}+git${EXT_SHORT}"
STAGE_DIR="$UTLX_SRC_DIR/_wheel_stage"
DIST_DIR="$UTLX_SRC_DIR/dist"

rm -rf "$STAGE_DIR" "$DIST_DIR"
mkdir -p "$STAGE_DIR" "$DIST_DIR"

cp -r "$UTLX_SRC_DIR/python/utlx_plugin" "$STAGE_DIR/utlx_plugin"
cp -r "$UTLX_SRC_DIR/python/utlx" "$STAGE_DIR/utlx"
cp -r "$UTLX_SRC_DIR/tlx/language/tlx" "$STAGE_DIR/tlx"
rm -rf "$STAGE_DIR/tlx/tutorials"
cp "$UTLX_BUILD_DIR/lib/libutlx.so" "$STAGE_DIR/utlx_plugin/libutlx.so"

cat > "$STAGE_DIR/pyproject.toml" << TOML
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "utlx"
version = "$UTLX_VERSION"
description = "uTLX: Triton Language Extensions distributed as a Plugin"
requires-python = ">=3.9"

[tool.setuptools]
packages = ["utlx", "utlx_plugin", "utlx_plugin.compiler", "tlx", "tlx.compiler"]

[tool.setuptools.package-data]
utlx_plugin = ["libutlx.so"]
TOML

# setup.py to force platform-specific wheel tag (ships native .so)
cat > "$STAGE_DIR/setup.py" << 'SETUP'
from setuptools import setup
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(distclass=BinaryDistribution)
SETUP

setup_venv
log "Building uTLX wheel..."
python -m build --wheel --no-isolation --outdir "$DIST_DIR" "$STAGE_DIR"
rm -rf "$STAGE_DIR"

OUT_DIR="$WHEELS_DIR/dist"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/utlx-*.whl
cp "$DIST_DIR"/utlx-*.whl "$OUT_DIR/"

UTLX_WHL=$(ls "$OUT_DIR"/utlx-*.whl | head -1)
log "Done: $UTLX_WHL"

if [ "$PUBLISH" = "1" ]; then
    publish_wheel "$UTLX_WHL" "$(release_tag_from_wheel "$UTLX_WHL")"
fi
