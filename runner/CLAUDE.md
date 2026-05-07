# Runner

Launches uTLX kernels with the wheel's plugin loaded and a registry of monkey
patches that bridge `utlx_plugin`'s stale Python-side API to the currently-
installed Triton's C++ bindings.

## Files

- **`runner.py`** — sets `TRITON_PLUGIN_PATHS` so libtriton loads `libutlx.so`,
  imports triton + utlx, resolves the patch list for the kernel, applies
  patches, then `runpy`'s the kernel.
- **`tlx_patches.py`** — patch registry. Each patch is a function decorated
  with `@register("name", default=True)` carrying a docstring with the bridge
  rationale and a `Retire when:` hint.
- **`tlx_patches.toml`** — per-wheel patch selection keyed by utlx commit.

## Usage

```bash
python runner/runner.py kernels/<kernel>.py [kernel_args...]
```

The kernel can opt out of any subset of patches by declaring at top level:

```python
__tlx_patches__ = ["semantic_shims", "warp_specialize_codegen"]
# or:
__tlx_patches__ = "all"
```

## Why this directory exists

The pre-built `utlx_plugin` wheel was authored against an older Triton API. As
upstream Triton moves, the wheel's Python code drifts out of sync with the C++
bindings shipped by `libtriton`. Rather than re-cutting wheels for every API
churn, `tlx_patches.py` carries the bridge layer as a registry of selectable
patches.

When you cut a new wheel that obsoletes one or more patches, add an entry to
`tlx_patches.toml` keyed by the new commit and list only the patches still
needed. The old commit's entry stays so older wheels keep working.

**When to add a patch here vs. rebuild the wheel:**
- Add a patch if the fix is small (≲50 lines), Python-only, and unblocks a
  kernel today.
- Rebuild the wheel if the fix needs C++ (new pybind binding, new MLIR op
  builder), if the patch grows past ~100 lines, or if patches start
  interacting in fragile ways.

## Patch selection

`tlx_patches.resolve_for_kernel(path)` returns the patch list to apply, in
priority order:

1. Kernel module declares `__tlx_patches__ = [...]` (or `"all"`) at top
   level. Read via AST; the kernel is not executed for this lookup.
2. `tlx_patches.toml` section matching the installed utlx wheel commit, e.g.
   `[utlx."f3d635af"]` for `utlx-0.1.0+gitf3d635af`.
3. `tlx_patches.toml` `[default]` section.
4. All patches registered with `default=True`.

Order of registration in `tlx_patches.py` matches dependency order — when
adding a new patch, append it at the right spot. `apply()` always runs in
registration order regardless of the input list's order.

## Loaded plugin surface

`libutlx.so` only exports the `tritonGetPluginInfo()` C-API. That registers
passes, dialects, and "ops" (plugin op callbacks reachable on the builder as
`utlx_*` methods). Crucially, the pybind extensions in
`extensions/utlx/tlx/dialect/triton_tlx.cc` (`init_triton_tlx_ir`) are **never
loaded** — those defs (e.g. the 7-arg `create_make_tensor_descriptor` with
`desc_ptr`, `create_warp_specialize_op`) are dead code at runtime. If you hit
`'ir.builder' object has no attribute 'create_X'`, that's almost always why.

## Bridge categories

1. **Semantic-method shims** — restore methods that were on `TritonSemantic`
   in the older Triton (e.g. `_prepare_legacy_load`, `dot_precheck`).
2. **`visit_With` dispatch** — current Triton's `CodeGenerator.visit_With` has
   no extension hook, so uTLX's `TLX_WITH_DISPATCH` table is never consulted.
   We patch `visit_With` to consult it before the default flow, which routes
   `with tlx.async_tasks(): / with tlx.async_task(...):` to the uTLX codegen.
3. **TLX builtin signature shims** — JIT wraps `None` literals into
   `constexpr(None)`; type checks that compare `is None` need an unwrap.
4. **GluonOpBuilder swap** — most TLX-relevant ops (`create_warpgroup_mma`,
   `create_async_tma_copy_*`, `create_local_alloc`, `create_memdesc_index`,
   `create_warp_specialize`, …) are bound only on `gluon_ir.GluonOpBuilder`.
   Since `GluonOpBuilder` is a subclass of `ir.builder`, swap it in after
   `CodeGenerator.__init__` while keeping `TritonSemantic`. Where pybind
   method resolution then picks the wrong overload (e.g. gluon's 5-arg
   `create_make_tensor_descriptor` vs. the regular 6-arg form), call the
   regular `ir.builder.create_X` unbound method explicitly to bypass.
5. **`create_warpgroup_mma` `useAcc` default** — uTLX passes `None`; the
   binding now requires an `ir.value`. Wrap the method to default to
   `get_int1(True)`.
6. **WarpSpecializeOp codegen rewrite** — the legacy uTLX
   `visit_withAsyncTasks` targets a stale IR shape (flat partition regions on
   the WS op + `append_operand` for captures). Current Triton's
   `WarpSpecializeOp` has only `defaultRegion` + `partitionOpHolder`; the N
   partition regions live inside a nested `WarpSpecializePartitionsOp` whose
   `explicitCaptures` operand carries the captures. Reimplement the codegen
   along the lines of `triton/python/triton/experimental/gluon/language/_semantic.py`
   `warp_specialize`, using a `GluonOpBuilder` that shares the `MLIRContext`
   for the WS structural ops.

## Diagnostic recipe

When a kernel breaks after a Triton bump:
```bash
python runner/runner.py kernels/<kernel>.py 2>&1 | tail -40
```
Common error shapes and what they mean:
- `'ir.builder' object has no attribute 'create_X'` → op is bound only on
  `GluonOpBuilder`; ensure the GluonOpBuilder swap is active, or wrap the
  call to use one explicitly.
- `incompatible function arguments. The following argument types are
  supported: 1. (...)` → signature drift; either patch the call site or
  monkey-patch the bound method.
- `Did you forget to add @triton.jit ?` → a `@tl.builtin` is being called
  from host code; switch to the host-side equivalent (e.g.
  `triton.tools.tensor_descriptor.TensorDescriptor.from_tensor`).
- `desc_ptr must be None or tlx.tensor_descriptor_ptr, got
  <class 'triton.language.core.constexpr'>` → the constexpr unwrap shim is
  missing; arg defaulted to `None` got wrapped by the JIT.
