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
- **`make_submission.py`** — generates a self-contained submission by combining `install_deps.py`, `runner/runner.py` (setup), `runner/tlx_patches.py` (patch registry), and the user's kernel file.

## Generate submission

```bash
python make_submission.py submission.py              # prints to stdout
python make_submission.py submission.py -o out.py    # writes to file
```

The generated file is self-contained: installs wheels, sets up the uTLX plugin, applies the resolved set of monkey-patches, then runs the kernel. Source files are read at generation time — they are the single source of truth.

## Monkey-patches

The submission embeds the entire `runner/tlx_patches.py` registry plus a hardcoded `apply([...])` call with the resolved patch list. Resolution happens **at generation time** (not at runtime — the GPUMode container installs a wheel whose commit may not match the local `tlx_patches.toml` entries):

1. If the kernel file declares `__tlx_patches__ = [...]` (or `"all"`) at top level, use that list.
2. Otherwise apply all patches registered with `default=True` in `tlx_patches.py`.

To customize per-kernel, add the declaration at the top of your kernel file:

```python
__tlx_patches__ = ["semantic_shims", "warp_specialize_codegen"]
```

See `../runner/tlx_patches.py` and the project-root `CLAUDE.md` "API Bridging" section for the full patch list and rationale for each.
