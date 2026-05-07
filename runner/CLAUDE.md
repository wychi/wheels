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

## Patch catalog

Each registered patch fixes a specific failure mode. Full rationale and
`Retire when:` notes live in each patch's docstring in `tlx_patches.py`.
For the proper-fix-location classification (`utlx-py` / `utlx-cpp` / `triton`)
and key files to touch in a wheel rebuild, see [`ENABLEMENT.md` →
"Patch Follow-ups"](../ENABLEMENT.md#patch-follow-ups-where-the-proper-fix-belongs).

| Patch                          | Symptom (without it)                                                                                                   | Triggered by kernel construct          | Fix lives in |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------|--------------|
| `semantic_shims`               | `'TritonSemantic' object has no attribute '_prepare_legacy_load'` (or `dot_precheck`, or `tl._unwrap_if_constexpr`)    | Any `tlx.async_load` or `tlx.async_dot`| utlx-py      |
| `dispatch_visit_with`          | `with tlx.async_tasks():` blocks fall through to default codegen (silent — produces wrong IR)                          | `with tlx.async_tasks(): / async_task` | triton       |
| `make_tensor_descriptor`       | `desc_ptr must be None or tlx.tensor_descriptor_ptr, got <class 'triton.language.core.constexpr'>`, or 5-vs-6-arg overload mismatch | `tlx.make_tensor_descriptor`  | utlx-py      |
| `wgmma_use_acc_default`        | `create_warpgroup_mma(): incompatible function arguments` — `useAcc=None` rejected                                     | `tlx.async_dot`                        | utlx-py      |
| `broadcast_shape_overload`     | `create_broadcast(): incompatible function arguments` — gluon overload takes `ir.type`, semantic passes shape list     | Any 2D-tensor expression (broadcasting)| triton (or disappears with the swap) |
| `gluon_op_builder_swap`        | `'ir.builder' object has no attribute 'create_warpgroup_mma'` (or `create_local_alloc`, `create_memdesc_index`, `create_warp_specialize`, …) | `tlx.local_alloc`, `tlx.async_dot`, `tlx.async_tasks` | utlx-py |
| `async_load_native`            | `'ttg.async_copy_global_to_local' op operand count (3) does not match with the total size (0) specified in attribute 'operandSegmentSizes'` | `tlx.async_load`           | utlx-py *or* utlx-cpp |
| `wgmma_acc_layout_setup`       | `failed to legalize unresolved materialization from ('tensor<…,#blocked>') to ('tensor<…>')` in `TLXConvertTritonToTritonGPU` | `tlx.async_dot`                | utlx-cpp (partial — see below) |
| `warp_specialize_codegen`      | `'WarpSpecializeOp' object has no attribute 'get_partition_region'` / `'append_operand'`                               | `with tlx.async_tasks(): / async_task` | utlx-py      |

> **Outstanding wall:** Even with all patches active, `tiny_gemm.py` fails at
> `TritonGPURemoveLayoutConversions` due to the output-side
> `tlx.release_layout` marker — `wgmma_acc_layout_setup` only handles the
> input side. See [`ENABLEMENT.md` → 47debefa](../ENABLEMENT.md#47debefa) for the
> bisect run that established the minimum patch set, and the
> [`utlx-cpp`](../ENABLEMENT.md#utlx-cpp--fix-in-next-wheel-rebuild-c) /
> [`utlx-py`](../ENABLEMENT.md#utlx-py--fix-in-next-wheel-rebuild-python-only)
> tables for fix paths.

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
