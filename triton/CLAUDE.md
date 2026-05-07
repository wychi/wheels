# Triton Wheels

Patched Triton wheels with uTLX plugin support.

## Patches

1. **Build flag** — built with `-DTRITON_EXT_ENABLED=ON` to enable plugin loading (default is OFF).

2. **`python/src/ir.cc`** (release/3.7.x only) — fix plugin op builder to support return values. Already upstreamed in Triton `main` as of `7cff1f27`.

## Build

```bash
./build_triton_wheel.sh <triton-commit>
```

The script handles the full dependency chain (LLVM checkout/build + Triton wheel build). Output goes to `../dist/`.

See `../env.sh` for configurable defaults (`CUDA_HOPPER`, `CUDA_BLACKWELL`, `PYTHON_VERSION`, etc.).

## Notes

- Requires `cmake>=3.20,<4.0` (cmake 4.x is incompatible)
- `env.sh` handles `TRITON_CACHE_PATH`, clang++ symlink, and CUDA paths automatically
