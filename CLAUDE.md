# Wheels

Pre-built Python wheels for GPU kernel development with uTLX (Triton Language Extensions).

## Folder Structure

```
wheels/
├── dist/                    # Built wheels (all versions, flat)
├── triton/                  # Triton build script + docs
├── utlx/                    # uTLX build script + docs
├── kernels/                 # Example uTLX kernels
│   └── tiny_gemm.py
├── runner/                  # Kernel runner + API-bridge patches
│   ├── runner.py            # uTLX setup + applies tlx_patches, runs kernel
│   ├── tlx_patches.py       # Registry of monkey patches bridging API drift
│   └── tlx_patches.toml     # Patch selection per utlx wheel commit
├── gpumode/                 # GPUMode competition bundle
│   ├── install_deps.py      # Wheel URLs + install function
│   └── make_submission.py   # Generate self-contained submission
├── env.sh                   # Shared build helpers
├── release.sh               # End-to-end build + test + publish
└── test_runner.py           # Install wheels + run utlx core tests
```

## Quick Start

```bash
# Build + test everything from a triton-ext commit
./release.sh <triton-ext-commit>

# Build + test + publish to GitHub Releases
./release.sh <triton-ext-commit> --publish

# Just run tests (installs latest wheels from dist/)
python test_runner.py

# Run a uTLX kernel (assumes wheels already installed)
python runner/runner.py kernels/tiny_gemm.py

# Generate a self-contained GPUMode submission
python gpumode/make_submission.py submission.py -o submission_tlx.py
```

## Dependency Chain

```
triton-ext (uTLX)
  │  pins Triton commit in: ci/triton-hash.txt
  ▼
Triton
  │  pins LLVM commit in: cmake/llvm-hash.txt
  ▼
LLVM/MLIR
```

To find compatible versions, start from triton-ext and follow the pins:

```bash
cat ~/oss/triton-ext/ci/triton-hash.txt        # → Triton commit
git -C ~/oss/triton show <commit>:cmake/llvm-hash.txt  # → LLVM commit
```

The pin files are the single source of truth. To override a dependency, update the pin and commit — don't use build flags or env vars:

```bash
# Override Triton version
cd ~/oss/triton-ext
echo "<triton-hash>" > ci/triton-hash.txt
git commit -am "ci: update Triton pin to <short-hash>"

# Override LLVM version
cd ~/oss/triton
echo "<llvm-hash>" > cmake/llvm-hash.txt
git commit -am "Update LLVM pin to <short-hash>"
```

## Source Repositories

| Repo | Local Path | Upstream |
|------|-----------|----------|
| Triton | `~/oss/triton` | `triton-lang/triton` |
| triton-ext | `~/oss/triton-ext` | — |
| LLVM | `~/oss/llvm-project` | `llvm/llvm-project` |

## Build Order

Build bottom-up (each step needs the one below it):

1. **LLVM** — `cmake -S llvm -B build -G Ninja` (~30-60 min)
2. **Triton** — `python -m build --wheel --no-isolation` (~15-20 min, needs LLVM)
3. **uTLX** — `cmake + ninja` then wheel packaging (~5 min, needs LLVM + Triton)

## Build Scripts

- **`release.sh <triton-ext-commit>`** — full pipeline: resolves deps, builds everything, runs tests, optionally publishes
- **`triton/build_triton_wheel.sh <triton-commit>`** — builds Triton wheel (handles LLVM automatically)
- **`utlx/build_utlx_wheel.sh <triton-ext-commit>`** — builds uTLX wheel (requires Triton already built)
- **`env.sh`** — shared helpers: `resolve_deps`, `setup_triton_env`, `setup_venv`, `setup_llvm_tools`, etc.

## Build Dependencies

- **Python** — configurable via `PYTHON_VERSION` (default: 3.13), install via `uv python install`
- **LLVM** — built from source (projects `mlir;lld;clang`, targets `host;NVPTX;AMDGPU`)
- **CUDA** — ptxas 12.8 (Hopper), ptxas 13.1 (Blackwell)
- **Build tools** — `build`, `cmake<4.0`, `ninja`, `setuptools`, `wheel`, `pybind11`

## Build Notes

- Use `--no-isolation` with `python -m build` (devserver blocks `pip`, use `uv pip`)
- Set `TRITON_BUILD_UT=OFF` (avoids googletest download, network restricted)
- Set `TRITON_BUILD_PROTON=OFF` (skip Proton build)

## API Bridging (Monkey Patching)

The pre-built `utlx_plugin` wheel was authored against an older Triton API and
needs Python-side bridging to the currently-installed Triton's C++ bindings.
That bridge layer lives in `runner/` as a registry of selectable patches —
see [`runner/CLAUDE.md`](runner/CLAUDE.md) for the patch catalog, selection
rules, and diagnostic recipes.

## uTLX Enablement

For commit-specific bug fixes, patch requirements, and enablement work, see
[`ENABLEMENT.md`](ENABLEMENT.md). That document tracks issues and
fixes organized by utlx wheel commit.

To enable a new kernel against the current wheel — or hand the task to
another agent — invoke the `enable-kernel` skill (auto-triggers on
phrases like "enable this kernel" / "make this kernel work"). The full
workflow lives in
[`.claude/skills/enable-kernel/SKILL.md`](.claude/skills/enable-kernel/SKILL.md).

## Testing

```bash
python test_runner.py           # install latest wheels from dist/ + run core tests
python test_runner.py --all     # run all tests (some may segfault)
python test_runner.py -k test_barriers  # filter tests
```

Core tests: barriers, local_alloc, mem_ops, storage_alias, types, custom_stages (128 tests).

## Wheel Naming

Both wheels embed the source commit in the version's local segment:

- `triton-<version>+git<commit>-cp313-cp313-linux_x86_64.whl`
- `utlx-<version>+git<commit>-cp313-cp313-linux_x86_64.whl`

## Hosting

Wheels are published as GitHub Releases on `wychi/wheels`. Each wheel gets its own release tag:
- `triton-<version>-<commit>`
- `utlx-<version>-<commit>`
