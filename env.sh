#!/usr/bin/env bash
# env.sh — Shared helpers for wheel build scripts.
#
# Source this, then use the functions:
#   source env.sh
#   log "hello"
#   setup_triton_env /path/to/llvm/build

# ── defaults ──────────────────────────────────────────────────────────────────

: "${CUDA_HOPPER:=/usr/local/cuda-12.8}"
: "${CUDA_BLACKWELL:=/usr/local/cuda-13.1}"
: "${TRITON_REPO:=$HOME/oss/triton}"
: "${TRITON_EXT_REPO:=$HOME/oss/triton-ext}"
: "${LLVM_REPO:=$HOME/oss/llvm-project}"
: "${PYTHON_VERSION:=3.13}"
: "${REPO:=wychi/wheels}"

# ── common helpers ───────────────────────────────────────────────────────────

log() { echo "==> $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }

check_dir() { [ -d "$1" ] || err "$1 does not exist"; }

get_short_hash() { git -C "$1" rev-parse --short=8 "$2"; }

# ── resolve_deps ─────────────────────────────────────────────────────────────
# Given a triton-ext commit, resolves the full dependency chain.
# Sets: EXT_FULL, EXT_SHORT, TRITON_PIN, TRITON_SHORT, LLVM_PIN, LLVM_SHORT

resolve_deps() {
    local ext_commit="${1:?Usage: resolve_deps <triton-ext-commit>}"

    EXT_FULL=$(git -C "$TRITON_EXT_REPO" rev-parse "$ext_commit") \
        || err "Cannot resolve '$ext_commit' in $TRITON_EXT_REPO"
    EXT_SHORT=$(get_short_hash "$TRITON_EXT_REPO" "$EXT_FULL")

    TRITON_PIN=$(git -C "$TRITON_EXT_REPO" show "$EXT_FULL:ci/triton-hash.txt" | tr -d '[:space:]') \
        || err "Cannot read ci/triton-hash.txt from triton-ext $EXT_SHORT"
    TRITON_SHORT=$(get_short_hash "$TRITON_REPO" "$TRITON_PIN")

    LLVM_PIN=$(git -C "$TRITON_REPO" show "$TRITON_PIN:cmake/llvm-hash.txt" | tr -d '[:space:]') \
        || err "Cannot read cmake/llvm-hash.txt from Triton $TRITON_SHORT"
    LLVM_SHORT=$(get_short_hash "$LLVM_REPO" "$LLVM_PIN")

    log "triton-ext: $EXT_SHORT  Triton: $TRITON_SHORT  LLVM: $LLVM_SHORT"
}

# ── setup_llvm_tools ─────────────────────────────────────────────────────────
# Ensures clang++ is available in the LLVM build dir (symlinks system if needed).

setup_llvm_tools() {
    local llvm_build="${1:?Usage: setup_llvm_tools <llvm-build-dir>}"

    [ -f "$llvm_build/bin/mlir-tblgen" ] || err "mlir-tblgen not found in $llvm_build/bin"

    if [ ! -f "$llvm_build/bin/clang++" ]; then
        local sys_clangxx
        sys_clangxx=$(which clang++ 2>/dev/null || true)
        [ -n "$sys_clangxx" ] || err "clang++ not found. Install clang or build LLVM with clang project."
        ln -sf "$sys_clangxx" "$llvm_build/bin/clang++"
        log "Symlinked clang++ -> $sys_clangxx"
    fi
}

# ── setup_triton_env ─────────────────────────────────────────────────────────
# Exports all env vars needed to build a Triton wheel.

setup_triton_env() {
    local llvm_build="${1:?Usage: setup_triton_env <llvm-build-dir>}"
    local hopper="${CUDA_HOPPER}"
    local blackwell="${CUDA_BLACKWELL}"

    [ -f "$hopper/bin/ptxas" ] || err "Hopper ptxas not found at $hopper/bin/ptxas"
    [ -f "$blackwell/bin/ptxas" ] || err "Blackwell ptxas not found at $blackwell/bin/ptxas"

    export LLVM_SYSPATH="$llvm_build"
    export TRITON_PTXAS_PATH="$hopper/bin/ptxas"
    export TRITON_PTXAS_BLACKWELL_PATH="$blackwell/bin/ptxas"
    export TRITON_CUOBJDUMP_PATH="$blackwell/bin/cuobjdump"
    export TRITON_NVDISASM_PATH="$blackwell/bin/nvdisasm"
    export TRITON_CUDACRT_PATH="$blackwell/"
    export TRITON_CUDART_PATH="$blackwell/"
    export TRITON_CUPTI_INCLUDE_PATH="$hopper/include"
    export TRITON_CUPTI_LIB_PATH="$hopper/lib64"
    export TRITON_CUPTI_LIB_BLACKWELL_PATH="$blackwell/lib64"
    export TRITON_CUPTI_PATH="$hopper/"
    export TRITON_BUILD_UT=OFF
    export TRITON_BUILD_PROTON=OFF
    export TRITON_EXT_ENABLED=ON
    export TRITON_CACHE_PATH="${TRITON_CACHE_PATH:-$HOME/.triton}"
    export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF -DTRITON_EXT_ENABLED=ON -DCMAKE_CXX_FLAGS=-Wno-attributes"

    log "Triton env configured (LLVM=$llvm_build)"
}

# ── setup_venv ───────────────────────────────────────────────────────────────
# Activates (or creates) the wheels project venv at <wheels>/.venv. Sets
# PYTHON and VENV_DIR. Build and test share this single venv.

WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

setup_venv() {
    VENV_DIR="$WHEELS_DIR/.venv"

    if [ -n "${PYTHON:-}" ]; then
        log "Using provided PYTHON=$PYTHON"
    else
        if [ ! -f "$VENV_DIR/bin/python" ]; then
            log "Creating Python $PYTHON_VERSION venv at $VENV_DIR..."
            uv python install "$PYTHON_VERSION"
            uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
        fi
        PYTHON="$VENV_DIR/bin/python"
        # Idempotent: ensures build deps are present even if venv pre-existed.
        uv pip install --python "$PYTHON" --quiet \
            build 'cmake>=3.20,<4.0' ninja setuptools wheel pybind11
    fi

    source "$VENV_DIR/bin/activate"
}

# ── find_libtriton ───────────────────────────────────────────────────────────
# Finds libtriton.so from Triton build output. Sets LIBTRITON, TRITON_CMAKE_DIR.

find_libtriton() {
    local py_short
    py_short=$(echo "$PYTHON_VERSION" | tr -d '.')

    TRITON_CMAKE_DIR="$TRITON_REPO/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}"
    LIBTRITON=""

    for candidate in \
        "$TRITON_REPO/build/lib.linux-x86_64-cpython-${py_short}/triton/_C/libtriton.so" \
        "$TRITON_REPO/python/triton/_C/libtriton.so"; do
        if [ -f "$candidate" ]; then
            LIBTRITON="$candidate"
            break
        fi
    done

    [ -n "$LIBTRITON" ] || err "libtriton.so not found. Build Triton first."
}

# ── read_triton_version ──────────────────────────────────────────────────────
# Reads TRITON_VERSION from setup.py at a given commit (or HEAD).

read_triton_version() {
    local commit="${1:-HEAD}"
    TRITON_VERSION=$(git -C "$TRITON_REPO" show "$commit:setup.py" \
        | grep -oP 'TRITON_VERSION\s*=\s*"\K[^"]+' || echo "3.7.0")
}

# ── publish ─────────────────────────────────────────────────────────────────
# Uploads a wheel to GitHub Releases. Creates the release if it doesn't exist.

publish_wheel() {
    local whl="$1" tag="$2" name
    name="$(basename "$whl")"
    # Route gh's stdout to stderr so the only thing on this function's stdout
    # is the download URL — callers capture that with $(...).
    if gh release view "$tag" --repo "$REPO" >&2 2>/dev/null; then
        gh release upload "$tag" "$whl" --repo "$REPO" --clobber >&2
    else
        gh release create "$tag" "$whl" --repo "$REPO" --title "$tag" --notes "Wheel: $name" >&2
    fi
    echo "https://github.com/$REPO/releases/download/$tag/$name"
}

# ── release_tag_from_wheel ──────────────────────────────────────────────────
# Derives a release tag from a wheel filename.
#   triton-3.7.0+gitca6bd0c3-cp313-cp313-linux_x86_64.whl → triton-3.7.0-ca6bd0c3
#   utlx-0.1.0+gitd4769dbb.patched-cp313-cp313-linux_x86_64.whl → utlx-0.1.0-d4769dbb.patched

release_tag_from_wheel() {
    local name
    name="$(basename "$1")"
    echo "$name" | sed -E 's/^([a-z]+)-([0-9.]+)\+git([a-f0-9.]+[a-z]*)-.*$/\1-\2-\3/'
}
