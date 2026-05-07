# Wheels

Pre-built Triton + uTLX (Triton Language Extensions) wheels for GPU kernel development.

## Get Started

```bash
uv pip install "triton @ https://github.com/wychi/wheels/releases/download/<tag>/<triton-wheel>"
uv pip install "utlx @ https://github.com/wychi/wheels/releases/download/<tag>/<utlx-wheel>"
export TRITON_PLUGIN_PATHS=$(python -c 'import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), "libutlx.so"))')
python kernels/tiny_gemm.py
```

## Build

Start from a triton-ext commit — the build scripts resolve the full dependency chain automatically:

```bash
# Build + test both wheels
./build_wheels.sh <triton-ext-commit>

# Publish the resulting dist/*.whl to GitHub Releases
./gh_release.sh
```

Or build individually:

```bash
# Triton wheel (builds LLVM if needed, ~30-60 min first time)
./triton/build_triton_wheel.sh <triton-commit>

# uTLX wheel (requires Triton already built)
./utlx/build_utlx_wheel.sh <triton-ext-commit>
```

Output goes to `dist/`.

## Install

```bash
pip install dist/triton-*.whl
pip install dist/utlx-*.whl
```

Set the plugin path before importing Triton:

```bash
export TRITON_PLUGIN_PATHS=$(python -c 'import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), "libutlx.so"))')
```

Or install from GitHub Releases:

```bash
pip install "triton @ https://github.com/wychi/wheels/releases/download/<tag>/<wheel>"
pip install "utlx @ https://github.com/wychi/wheels/releases/download/<tag>/<wheel>"
```

## Test

```bash
python test_runner.py              # core tests (128 tests)
python test_runner.py --all        # all tests
python test_runner.py -k <pattern> # filter tests
```

## Wheel Naming

Both wheels embed the source commit in the version's local segment:

- `triton-<version>+git<commit>-cp313-cp313-linux_x86_64.whl`
- `utlx-<version>+git<commit>-cp313-cp313-linux_x86_64.whl`
