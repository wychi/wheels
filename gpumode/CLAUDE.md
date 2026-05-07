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

## Usage in submission.py

### 1. Install wheels

```python
TRITON_WHEEL_URL = "https://github.com/wychi/wheels/releases/download/<tag>/triton-*.whl"
UTLX_WHEEL_URL = "https://github.com/wychi/wheels/releases/download/<tag>/utlx-*.whl"

subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                f"triton @ {TRITON_WHEEL_URL}",
                f"utlx @ {UTLX_WHEEL_URL}"])
```

### 2. Set up uTLX plugin path

```python
import os, sysconfig
os.environ["TRITON_CACHE_PATH"] = os.path.expanduser("~/.triton")

dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
```

Both must be set **before** `import triton`.

### 3. Import and use

```python
import triton
import triton.language as tl
import utlx_plugin as tlx
```

### 4. Monkey-patches

Some `triton.language` functions need patching for uTLX compatibility (see `submission.py` for `apply_tlx_patches()`). These patches provide `_unwrap_if_constexpr`, `_prepare_legacy_load`, `dot_precheck`, and `create_warpgroup_mma`.

### 5. Generate submission

```bash
./make_submission.sh submission.py -o submission_tlx.py
```

This wraps a clean kernel file with install, plugin setup, and monkey-patch boilerplate.
