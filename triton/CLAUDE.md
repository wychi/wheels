# GPUMode Wheels

Pre-built Python wheels for the GPUMode competition environment.

## Environment

The target runtime is a Modal container:
- **Python**: 3.13
- **CUDA**: 12.9.1 (devel, ubuntu24.04)
- **PyTorch**: 2.11.0 (cu129)
- **Package manager**: `uv` (via `.uv_pip_install`)

## Triton Wheel

The official `triton==3.7.0` PyPI wheel has a bug that breaks uTLX plugin support. This project builds a patched Triton wheel from the `release/3.7.x` branch of `triton-lang/triton` with two fixes applied:

1. `CMakeLists.txt` — add `-Wno-attributes` to suppress build warnings
2. `python/src/ir.cc` — fix plugin op builder to support return values (required for uTLX)

## Build Dependencies

- **Python 3.13** — install via `uv python install 3.13`
- **LLVM** — built from source at the commit specified in `triton/cmake/llvm-hash.txt`, with projects `mlir;lld` and targets `host;NVPTX;AMDGPU`
- **CUDA toolchain** — ptxas 12.8 (Hopper), ptxas 13.1 (Blackwell), installed via `feature install cuda_13_1`
- **Build tools** — `build`, `cmake`, `ninja`, `setuptools`, `wheel`, `pybind11` (install via `uv pip install`)

## Build Notes

- Use `--no-isolation` with `python -m build` because the devserver blocks `pip` (use `uv pip` instead)
- Set `TRITON_BUILD_UT=OFF` to avoid googletest download (network restricted)
- Set `TRITON_BUILD_PROTON=OFF` to skip Proton build
- Set `LLVM_SYSPATH` to point to the local LLVM build directory
- Set `TRITON_PTXAS_PATH`, `TRITON_PTXAS_BLACKWELL_PATH`, etc. to use local CUDA installations instead of downloading from NVIDIA
- The full build script is in `README.md`

## Hosting

Wheels are uploaded as GitHub Releases on `wychi/wheels` since they are too large (~327MB) for regular git.
