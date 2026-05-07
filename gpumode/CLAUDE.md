# GPUMode

Competition bundle for [GPUMode](https://gpumode.com/).

## Runtime Environment (Modal container)

- **Python**: 3.13
- **CUDA**: 12.9.1 (devel, ubuntu24.04)
- **PyTorch**: 2.11.0 (cu129)
- **Package manager**: `uv` (via `.uv_pip_install`)

## Required Wheels

Install from `dist/` (or from GitHub Releases):

- **Triton** — patched with `TRITON_EXT_ENABLED=ON` for uTLX plugin support
- **uTLX** — Triton Language Extensions plugin

## Files

- **`install_deps.py`** — wheel URLs and `_install_custom_deps()` function. Edit this file to change wheel versions.
- **`make_submission.py`** — generates a self-contained submission by combining `install_deps.py`, `kernels/runner.py` (setup + patches), and the user's kernel file.

## Generate submission

```bash
python make_submission.py submission.py              # prints to stdout
python make_submission.py submission.py -o out.py    # writes to file
```

The generated file is self-contained: installs wheels, sets up the uTLX plugin, applies monkey-patches, then runs the kernel. The setup code is read from `install_deps.py` and `kernels/runner.py` at generation time — those files are the single source of truth.

## Monkey-patches

Some `triton.language` functions need patching for uTLX compatibility (defined in `kernels/runner.py`). These patches provide `_unwrap_if_constexpr`, `_prepare_legacy_load`, and `dot_precheck`.
