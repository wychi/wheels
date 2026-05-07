# uTLX Wheels

Triton Language Extensions plugin. Bundles `libutlx.so` (native plugin) + Python DSL packages.

## Build

```bash
./build_utlx_wheel.sh <triton-ext-commit>
```

Prerequisites: LLVM and Triton must already be built at the correct commits. The script verifies alignment and errors with instructions if not.

Output goes to `../dist/`.

## What's in the Wheel

| Package | Description |
|---------|-------------|
| `utlx_plugin` | Authoritative DSL + `libutlx.so` |
| `utlx_plugin.compiler` | Compiler integration |
| `utlx` | Legacy re-export shim |
| `tlx` | TLX language operators |
| `tlx.compiler` | TLX compiler |

## Runtime

```bash
export TRITON_CACHE_PATH=~/.triton
export TRITON_PLUGIN_PATHS=$(python -c 'import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), "libutlx.so"))')
```

Both must be set **before** `import triton`.

## Enablement

For commit-specific bugs, fixes, and patch requirements, see
[`../ENABLEMENT.md`](../ENABLEMENT.md).

## Known Test Issues

- `test_tlx.py::test_clock64` — segfaults, excluded from core test suite
- `test_tlx.py::test_async_tasks` — `_semantic` kwarg incompatibility
