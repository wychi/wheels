#!/usr/bin/env bash
# build_triton_wheel.sh — Build a Triton wheel from a specific commit.
#
# Usage:
#   ./build_triton_wheel.sh <triton-commit>
#   ./build_triton_wheel.sh <triton-commit> --publish
#
# Output:
#   dist/triton-*.whl

set -euo pipefail

TRITON_COMMIT="${1:?Usage: $0 <triton-commit> [--publish]}"
PUBLISH=0
[ "${2:-}" = "--publish" ] && PUBLISH=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$(dirname "$SCRIPT_DIR")"
LLVM_JOBS="${LLVM_JOBS:-$(nproc)}"
SKIP_LLVM_BUILD="${SKIP_LLVM_BUILD:-0}"

source "$WHEELS_DIR/env.sh"

# ── resolve triton commit ────────────────────────────────────────────────────

TRITON_FULL=$(git -C "$TRITON_REPO" rev-parse "$TRITON_COMMIT") \
    || err "Cannot resolve '$TRITON_COMMIT'"
TRITON_SHORT=$(get_short_hash "$TRITON_REPO" "$TRITON_FULL")
log "Triton: $TRITON_SHORT ($TRITON_FULL)"

LLVM_HASH=$(git -C "$TRITON_REPO" show "$TRITON_FULL:cmake/llvm-hash.txt" | tr -d '[:space:]')
LLVM_SHORT=$(get_short_hash "$LLVM_REPO" "$LLVM_HASH")
log "LLVM: $LLVM_SHORT ($LLVM_HASH)"

# ── build LLVM if needed ─────────────────────────────────────────────────────

LLVM_BUILD_DIR="$LLVM_REPO/build"

if [ "$SKIP_LLVM_BUILD" = "1" ]; then
    log "Skipping LLVM build (SKIP_LLVM_BUILD=1)"
else
    LLVM_CURRENT=$(git -C "$LLVM_REPO" rev-parse HEAD 2>/dev/null || echo "none")
    if [ "$LLVM_CURRENT" = "$LLVM_HASH" ] && [ -f "$LLVM_BUILD_DIR/bin/mlir-tblgen" ]; then
        log "LLVM already built at $LLVM_SHORT, skipping"
    else
        log "Building LLVM at $LLVM_SHORT..."
        git -C "$LLVM_REPO" checkout "$LLVM_HASH"
        cmake -S "$LLVM_REPO/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_PROJECTS="mlir;lld;clang" \
            -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
            -DLLVM_ENABLE_ASSERTIONS=ON
        cmake --build "$LLVM_BUILD_DIR" -j"$LLVM_JOBS"
    fi
fi

setup_llvm_tools "$LLVM_BUILD_DIR"

# ── checkout + build ─────────────────────────────────────────────────────────

git -C "$TRITON_REPO" checkout "$TRITON_FULL"
setup_venv
setup_triton_env "$LLVM_BUILD_DIR"
read_triton_version "$TRITON_FULL"

DIST_DIR="$WHEELS_DIR/dist"
mkdir -p "$DIST_DIR"
rm -f "$DIST_DIR"/triton-*.whl

log "Building Triton wheel..."
cd "$TRITON_REPO"
rm -f dist/triton-*.whl
"$PYTHON" -m build --wheel --no-isolation

cp dist/triton-*.whl "$DIST_DIR/"

TRITON_WHL=$(ls "$DIST_DIR"/triton-*.whl | head -1)
log "Done: $TRITON_WHL"

if [ "$PUBLISH" = "1" ]; then
    publish_wheel "$TRITON_WHL" "$(release_tag_from_wheel "$TRITON_WHL")"
fi
