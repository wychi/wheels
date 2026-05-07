# uTLX Enablement

This document tracks uTLX enablement work — bugs found, fixes applied, and patches needed for each utlx wheel commit. Each section is keyed by the utlx commit (the local segment of the wheel version).

## Commit Index

- [`f3d635af`](#f3d635af) — Initial wheel, requires full bridge layer
- [`1d7b7482`](#1d7b7482) — Current wheel with Triton 3.7.0+git7cff1f27

---

## f3d635af

**Wheel:** `utlx-0.1.0+gitf3d635af-cp313-cp313-linux_x86_64.whl`  
**Triton:** 3.7.0+git7cff1f27  
**Status:** Requires all patches from `tlx_patches.py`

### Issues & Fixes

#### 1. libtriton Import Failure: `basic_string::_M_construct null not valid`

**Symptom:**
```
.venv/lib/python3.13/site-packages/triton/knobs.py:15: in <module>
    from triton._C.libtriton import getenv, getenv_bool
E   ImportError: basic_string::_M_construct null not valid
```

**Diagnosis:** A C++ static initializer in `libtriton` (likely a plugin-registration path that triton-ext extends) is constructing a `std::string` from a null `const char*`. Triton's pybind module fails to load before any Python-side patch can run.

**Fix:** Track down the offending op/dialect/pass registration in `triton-ext/extensions/utlx/` whose name or description string is `nullptr`, give it a real value, then rebuild the wheel.

**Status:** Fixed in this commit.

---

#### 2. Missing TritonSemantic Methods

**Symptom:**
```
AttributeError: 'TritonSemantic' object has no attribute '_prepare_legacy_load'
AttributeError: 'TritonSemantic' object has no attribute 'dot_precheck'
```

**Diagnosis:** Upstream Triton removed `_prepare_legacy_load` and `dot_precheck` from the public `TritonSemantic` API. uTLX still calls these methods in its mma/load paths.

**Fix:** Added `semantic_shims` patch that restores:
- `_prepare_legacy_load` — handles legacy load preparation with mask/other/boundary_check validation
- `dot_precheck` — validates dot product inputs and prepares accumulator
- `tl._unwrap_if_constexpr` — utility to unwrap constexpr values

**Retire when:** uTLX's mma/load paths stop relying on these specific methods.

---

#### 3. `with` Statement Dispatch Broken

**Symptom:**
```
with tlx.async_tasks():
    ...
```
The uTLX async task blocks are not recognized; code falls through to default handling.

**Diagnosis:** Upstream Triton's `CodeGenerator.visit_With` has no extension hook, so uTLX's `TLX_WITH_DISPATCH` table is never consulted. The dispatch mechanism that routes `tlx.async_tasks()` and `tlx.async_task()` to custom codegen is bypassed.

**Fix:** Added `dispatch_visit_with` patch that wraps `CodeGenerator.visit_With` to consult `TLX_WITH_DISPATCH` before the default flow.

**Retire when:** Upstream gains a public hook for `with`-statement dispatch.

---

#### 4. `make_tensor_descriptor` Type Check Failure

**Symptom:**
```
TypeError: desc_ptr must be None or tlx.tensor_descriptor_ptr, got <class 'triton.language.core.constexpr'>
```

**Diagnosis:** Two issues:
1. The JIT wraps literal `None` args into `constexpr(None)`; uTLX's `is None` type check rejects the constexpr wrapper.
2. With GluonOpBuilder swap active, pybind11 picks gluon's 5-arg `create_make_tensor_descriptor` (explicit result type) over the regular 6-arg form (block_shape + is_signed) that uTLX invokes.

**Fix:** Added `make_tensor_descriptor` patch that:
- Unwraps constexpr values before `is None` checks
- Calls regular `ir.builder.create_make_tensor_descriptor` unbound method explicitly to bypass gluon's override

**Retire when:** uTLX's `make_tensor_descriptor` is updated for the new Triton signature, AND the JIT's constexpr wrapping is handled inside.

---

#### 5. `create_warpgroup_mma` Missing `useAcc` Argument

**Symptom:**
```
TypeError: create_warpgroup_mma(): incompatible function arguments. The following argument types are supported:
    1. (self, arg0: ir.value, arg1: ir.value, arg2: ir.value, arg3: ir.value, arg4: triton::DotInputPrecision, arg5: int, arg6: bool) -> ir.value
```

**Diagnosis:** `tlx.async_dot` passes `None` for `useAcc`; the binding now requires a real `ir.value`.

**Fix:** Added `wgmma_use_acc_default` patch that wraps `GluonOpBuilder.create_warpgroup_mma` to default `useAcc=None` to `get_int1(True)`.

**Retire when:** uTLX's `mma_ops.async_dot` passes a proper `get_int1(...)` instead of `None`.

---

#### 6. `create_broadcast` Signature Mismatch

**Symptom:**
```
TypeError: create_broadcast(): incompatible function arguments. The following argument types are supported:
    1. (self, value: ir.value, type: ir.type) -> ir.value
```

**Diagnosis:** With the GluonOpBuilder swap active, pybind11 resolves `create_broadcast` to gluon's overload that takes an explicit `ir.type`. But Triton's `TritonSemantic.broadcast_impl_*` calls `builder.create_broadcast(handle, shape)` with a list of ints, matching the regular `ir.builder` form.

**Fix:** Added `broadcast_shape_overload` patch that dispatches list-shape calls to regular `ir.builder.create_broadcast` and passes `ir.type` calls through to gluon's native overload.

**Retire when:** GluonOpBuilder.create_broadcast accepts a shape list, or uTLX moves to gluon natively.

---

#### 7. Missing GluonOpBuilder Methods

**Symptom:**
```
AttributeError: 'ir.builder' object has no attribute 'create_warpgroup_mma'
AttributeError: 'ir.builder' object has no attribute 'create_async_tma_copy_*'
AttributeError: 'ir.builder' object has no attribute 'create_local_alloc'
AttributeError: 'ir.builder' object has no attribute 'create_memdesc_index'
AttributeError: 'ir.builder' object has no attribute 'create_warp_specialize'
```

**Diagnosis:** Most TLX-relevant ops are bound only on `GluonOpBuilder`, not on the regular `ir.builder` (TritonOpBuilder). uTLX was built against an older API where these were available on the base builder.

**Fix:** Added `gluon_op_builder_swap` patch that swaps `CodeGenerator.builder` to `GluonOpBuilder` after `__init__`, while keeping `TritonSemantic`. Since `GluonOpBuilder` is a subclass of `ir.builder`, regular `tl.*` ops keep working while exposing gluon-only `create_*` methods.

**Retire when:** uTLX moves to gluon natively (sets `JITFunction.is_gluon`), or upstream binds these ops on the regular `TritonOpBuilder`.

---

#### 8. WarpSpecializeOp IR Shape Mismatch

**Symptom:**
```
AttributeError: 'WarpSpecializeOp' object has no attribute 'get_partition_region'
AttributeError: 'WarpSpecializeOp' object has no attribute 'append_operand'
```

**Diagnosis:** The legacy uTLX `visit_withAsyncTasks` targets a stale IR shape (flat partition regions on the WS op + `append_operand` for captures). Current Triton's `WarpSpecializeOp` has only `defaultRegion` + `partitionOpHolder`; the N partition regions live inside a nested `WarpSpecializePartitionsOp` whose `explicitCaptures` operand carries the captures.

**Fix:** Added `warp_specialize_codegen` patch that reimplements the codegen along the lines of `triton/python/triton/experimental/gluon/language/_semantic.py` `warp_specialize`, using a `GluonOpBuilder` that shares the `MLIRContext` for the WS structural ops.

**Capture detection note:** The original uTLX codegen used `self.used_vars` (a custom CodeGenerator attribute) to narrow captures to only-used outer vars. Current Triton's CodeGenerator doesn't expose that, so we over-capture from all non-constexpr `liveins`. Unused block args are harmless (DCE), but missing captures would violate IsolatedFromAbove on the partitions op.

**Retire when:** uTLX's `code_generator.py:visit_withAsyncTasks` is rewritten upstream against the new IR shape.

---

### Patch Configuration

```toml
[utlx."f3d635af"]
patches = [
    "semantic_shims",
    "dispatch_visit_with",
    "make_tensor_descriptor",
    "wgmma_use_acc_default",
    "broadcast_shape_overload",
    "gluon_op_builder_swap",
    "warp_specialize_codegen",
]
```

All 7 patches are required for this wheel.

---

## 1d7b7482

**Wheel:** `utlx-0.1.0+git1d7b7482-cp313-cp313-linux_x86_64.whl`  
**Triton:** 3.7.0+git7cff1f27  
**Status:** TBD — needs testing against current patch set

### Build Info

Built from triton-ext commit `1d7b7482`. This is a newer wheel than `f3d635af`.

### Issues & Fixes

*To be documented as issues are discovered and fixed.*

### Testing Checklist

- [ ] Verify wheel installs correctly
- [ ] Run `python runner/runner.py kernels/tiny_gemm.py`
- [ ] Run `python test_runner.py` (core tests)
- [ ] Determine which patches from `f3d635af` are still needed
- [ ] Update `tlx_patches.toml` with commit-specific patch list

### Patch Configuration

```toml
# Add entry here once testing is complete
# [utlx."1d7b7482"]
# patches = [...]
```

---

## Enablement Workflow

### When Cutting a New Wheel

1. **Build the wheel:**
   ```bash
   ./release.sh <triton-ext-commit>
   ```

2. **Re-evaluate the patch list** — see the [Patch re-evaluation
   playbook](#patch-re-evaluation-playbook-run-on-every-new-wheel)
   above. Bisect each patch to find which ones the new wheel obsoletes;
   inheriting the previous commit's full list defeats the purpose of
   rebuilding.

3. **If the minimal list passes all tests:**
   - Add entry to `runner/tlx_patches.toml`:
     ```toml
     [utlx."<new-commit>"]
     patches = [
         # Only the patches that still failed something during bisect.
     ]
     ```

4. **If tests fail:**
   - Diagnose the failure using `runner/CLAUDE.md` diagnostic recipes
   - Determine if fix should be:
     - **Python patch** (add to `tlx_patches.py`): if small (≲50 lines), Python-only, unblocks immediately
     - **Wheel rebuild** (fix in triton-ext): if needs C++ (new pybind binding, MLIR op), patch grows too large, or patches interact fragily

5. **Document the fix:**
   - Add entry to this file under the new commit section
   - Include: symptom, diagnosis, fix, retire condition

### When to Patch vs. Rebuild

**Add a patch if:**
- Fix is small (≲50 lines)
- Python-only change
- Unblocks a kernel today

**Rebuild the wheel if:**
- Fix needs C++ (new pybind binding, new MLIR op builder)
- Patch grows past ~100 lines
- Patches start interacting in fragile ways

---

## Patch Follow-ups: Where the Proper Fix Belongs

Each patch in `runner/tlx_patches.py` is a Python-side bridge that papers
over an issue with the wheel or with Triton itself. To shrink the bridge
layer over time, classify each patch by where the *proper* fix lives:

| Category    | Where                                     | Cost                                |
|-------------|-------------------------------------------|-------------------------------------|
| `utlx-py`   | `triton-ext/extensions/utlx/.../*.py`     | Wheel rebuild, Python-only          |
| `utlx-cpp`  | `triton-ext/extensions/utlx/.../*.cc`     | Wheel rebuild, C++ + MLIR knowledge |
| `triton`    | Upstream `triton-lang/triton`             | External coordination               |

### `utlx-py` — fix in next wheel rebuild (Python-only)

Highest leverage; smallest cost. Each retires one or more patches.

| Patch                          | What to change in utlx                                           |
|--------------------------------|------------------------------------------------------------------|
| `gluon_op_builder_swap`        | At each call site that needs gluon-only ops (`create_warpgroup_mma`, `create_local_alloc`, `create_memdesc_index`, `create_warp_specialize`, …), construct a `GluonOpBuilder` ad-hoc instead of relying on a global swap. Eliminates the swap *and* its cascade fixups (`broadcast_shape_overload`, `make_tensor_descriptor`'s 6-arg bypass). |
| `semantic_shims`               | Stop calling removed `TritonSemantic` methods (`_prepare_legacy_load`, `dot_precheck`); reimplement inline in `mma_ops.py` / `mem_ops.py`. |
| `make_tensor_descriptor`       | Update `mem_ops.make_tensor_descriptor` to use the current 5-arg gluon binding; unwrap constexpr-`None` for `desc_ptr` inside. |
| `wgmma_use_acc_default`        | `mma_ops.async_dot` should pass `_semantic.builder.get_int1(True)` for `useAcc` instead of `None`. ~5-line fix. |
| `warp_specialize_codegen`      | Rewrite `compiler/code_generator.py:visit_withAsyncTasks` against the current `WarpSpecializeOp` IR shape (defaultRegion + partitionOpHolder + nested `WarpSpecializePartitionsOp` with `explicitCaptures`). |
| `async_load_native` (option a) | Drop the custom `utlx_async_load` op; have `mem_ops.async_load` call `create_async_copy_global_to_local` + `create_async_commit_group` + `create_async_wait_group` directly. |

### `utlx-cpp` — fix in next wheel rebuild (C++)

| Patch                          | What to change in utlx                                           |
|--------------------------------|------------------------------------------------------------------|
| `async_load_native` (option b) | Fix the C++ side of `utlx_async_load` to set `operandSegmentSizes` correctly when constructing `ttg.async_copy_global_to_local`. Option (a) above is simpler. |
| `wgmma_acc_layout_setup`       | The `tlx.require_layout` / `tlx.release_layout` markers need a real lowering. Either: (1) add a conversion pattern in `tlx-convert-triton-to-tritongpu` that absorbs them into surrounding `ttg.convert_layout` ops, or (2) drop the markers entirely from the `mma_ops.async_dot` codepath and emit `ttg.convert_layout` to/from the wgmma-encoded acc directly. The current Python patch is half of (2) — input side only; output side needs the same treatment plus python-level type plumbing. |

### `triton` — needs upstream change (or stays as patch indefinitely)

| Patch                          | What needs to land upstream                                      |
|--------------------------------|------------------------------------------------------------------|
| `dispatch_visit_with`          | A public extension hook for `with`-statement dispatch in `CodeGenerator.visit_With`, so plugins can register handlers without monkey-patching. Until then, the patch is the only option. |
| `broadcast_shape_overload`     | Disappears once `gluon_op_builder_swap` is retired (no swap → no overload conflict). If the swap stays, upstream Triton would need to make `GluonOpBuilder.create_broadcast` accept either a shape list or an `ir.type`. |

### Key files

Pointers for the next session — where to look when implementing each fix.
Paths under `~/.venv/lib/python*/site-packages/utlx_plugin/` are inside the
*installed wheel*; the source-of-truth lives at the same relative path under
`triton-ext/extensions/utlx/python/utlx_plugin/`.

**uTLX Python source (rebuild target for `utlx-py` cluster):**

| Subsystem                  | Files                                                                  | Patches it would retire                          |
|----------------------------|------------------------------------------------------------------------|--------------------------------------------------|
| MMA ops (`async_dot`)      | `utlx_plugin/mma_ops.py` — `async_dot`, `require_nv_mma_shared_layout`, `require_dot_operand_layout` | `wgmma_use_acc_default`, `gluon_op_builder_swap` (mma side), `semantic_shims` (`dot_precheck`) |
| Memory ops (`async_load`)  | `utlx_plugin/mem_ops.py` — `async_load`, `async_load_commit_group`, `async_load_wait_group`, `make_tensor_descriptor` | `async_load_native`, `make_tensor_descriptor`, `semantic_shims` (`_prepare_legacy_load`) |
| Code generator             | `utlx_plugin/compiler/code_generator.py` — `visit_withAsyncTasks`      | `warp_specialize_codegen`                        |
| Dispatch table             | `utlx_plugin/compiler/dispatch.py` — `TLX_WITH_DISPATCH`               | (consumed by `dispatch_visit_with`)              |
| Pipeline / custom stages   | `utlx_plugin/custom_stages.py` — `make_ttir_wrapper`, `make_llir_wrapper` | (where new ttg attrs / passes hook in)        |

**uTLX C++ source (rebuild target for `utlx-cpp` cluster):**

| Subsystem                              | Files                                                                       | Patches it would retire           |
|----------------------------------------|-----------------------------------------------------------------------------|-----------------------------------|
| pybind init (`triton_tlx`)             | `triton-ext/extensions/utlx/tlx/dialect/triton_tlx.cc`                      | (currently dead — see below)      |
| `tlx.require_layout` / `tlx.release_layout` lowering | `triton-ext/extensions/utlx/.../TLXConvertTritonToTritonGPU.cpp` | `wgmma_acc_layout_setup`          |
| `utlx_async_load` op                   | `triton-ext/extensions/utlx/.../UtlxAsyncLoadOp.cpp` (or wherever the op constructor sets `operandSegmentSizes`) | `async_load_native` (option b) |

> Note: `libutlx.so` ships only `tritonGetPluginInfo`; the `init_triton_tlx_ir`
> pybind extensions in `triton_tlx.cc` are never loaded at runtime. If the
> rebuild plan involves new pybind bindings, that load path needs fixing too.

**Reference implementations in upstream Triton (model after these):**

| Topic                          | File                                                                                                          | Use for                                              |
|--------------------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| Hopper wgmma layout selection  | `triton/tools/triton_to_gluon_translator/hopper_helpers.py:_mmav3_acc_layout`                                 | `wgmma_acc_layout_setup` rewrite                     |
| Default blocked layout         | `triton/tools/triton_to_gluon_translator/common_helpers.py:default_blocked_layout`                            | Anywhere we need a fallback distributed layout       |
| Gluon `warp_specialize` codegen| `triton/experimental/gluon/language/_semantic.py:warp_specialize`                                             | `warp_specialize_codegen` rewrite (already mirrored) |
| Gluon module attrs setup       | `triton/experimental/gluon/_runtime.py:GluonASTSource.make_ir` (`ttg.target` / `ttg.num-warps` / etc.)        | Already mirrored in `gluon_op_builder_swap`          |
| Native `async_load`            | `triton/experimental/gluon/language/nvidia/ampere/async_copy.py:async_load`                                   | `async_load_native` (already mirrored)               |
| Native `warpgroup_mma`         | `triton/experimental/gluon/language/nvidia/hopper/__init__.py:warpgroup_mma`                                  | Reference for the proper acc-layout protocol         |
| `tl.dot` translation           | `triton/tools/triton_to_gluon_translator/hopper_helpers.py:tl_dot_mmav3`                                      | End-to-end pattern for gluon-style dot               |

**Patch registry (this repo):**

- `runner/tlx_patches.py` — all patches with rationale and `Retire when:` notes.
- `runner/tlx_patches.toml` — per-wheel-commit patch selection.
- `runner/CLAUDE.md` — patch catalog, selection rules, diagnostic recipes.

### Recommended sequencing for next wheel

1. `utlx-py` cluster — biggest wins, smallest cost. Land in this order:
   1. Refactor gluon-op call sites to use ad-hoc `GluonOpBuilder` (kills 3 patches).
   2. Inline the removed-API semantics, fix `make_tensor_descriptor`, fix `wgmma_use_acc_default`.
   3. Rewrite `visit_withAsyncTasks`.
   4. Drop `utlx_async_load`; call gluon ops directly.
2. `utlx-cpp` cluster — `wgmma_acc_layout_setup` is the only blocker for `tiny_gemm`; without this, the kernel still won't run end-to-end.
3. `triton` cluster — file an upstream issue for the `visit_With` hook; otherwise live with the patch.

After step 1 + 2, only `dispatch_visit_with` (and possibly `broadcast_shape_overload`) should remain on the patch side.

### Patch re-evaluation playbook (run on every new wheel)

When a new wheel lands, do NOT just inherit the previous commit's patch
list. Re-evaluate each patch — some may be obsolete. Procedure:

1. **Install the new wheel** and confirm the version:
   ```bash
   pip show utlx | awk '/^Version/ {print $2}'
   # → 0.1.0+git<new-commit>
   ```

2. **Bisect the patch list.** For each patch (in registration order),
   try disabling it and re-running `kernels/tiny_gemm.py`:
   ```bash
   # Edit kernels/tiny_gemm.py to add at the top:
   __tlx_patches__ = [<full list minus the one being tested>]
   python runner/runner.py kernels/tiny_gemm.py
   ```
   - If it still passes → patch is **obsolete**, retire it.
   - If it fails → patch is still load-bearing. Note which symptom returns.

3. **Run the broader test suite** with the candidate minimal list:
   ```bash
   python test_runner.py
   ```

4. **For any retired patch**, set `default=False` in
   `runner/tlx_patches.py` (don't delete — older wheels may still need
   it). Update its docstring's `Retire when:` line to note the
   retiring-wheel commit.

5. **Add the new wheel's entry** to `runner/tlx_patches.toml`:
   ```toml
   [utlx."<new-commit>"]
   patches = [
       # Only the patches that still failed something during step 2.
   ]
   ```

6. **Document the diff** under a new `## <new-commit>` section in this
   file:
   - What's fixed in the wheel since the previous commit (which patches
     became obsolete and why).
   - Any new failures and the fixes added (whether new patches or
     wheel-side TODOs).
   - Links to the relevant issues in the "Patch Follow-ups" tables above.

7. **Update the commit index** at the top of this file with the new
   entry.

> **Why bisect each patch?** Patches accumulate over wheel versions and
> retiring stale ones keeps the bridge layer minimal. A patch that was
> needed for `f3d635af` may be exactly what `1d7b7482` fixed in C++ — but
> there's no signal that obviates it unless we test.

---

## Known Test Issues

These tests are excluded from the core test suite:

### `test_tlx.py::test_clock64`
- **Status:** Segfaults
- **Action:** Excluded from core test suite
- **Tracking:** Needs investigation

### `test_tlx.py::test_async_tasks`
- **Status:** `_semantic` kwarg incompatibility
- **Action:** Excluded from core test suite
- **Tracking:** Needs investigation

---

## Common Error Patterns

### `'ir.builder' object has no attribute 'create_X'`
- **Cause:** Op is bound only on `GluonOpBuilder`
- **Fix:** Ensure `gluon_op_builder_swap` patch is active, or wrap the call to use `GluonOpBuilder` explicitly

### `incompatible function arguments. The following argument types are supported: 1. (...)`
- **Cause:** Signature drift between uTLX expectations and current Triton bindings
- **Fix:** Either patch the call site or monkey-patch the bound method to adapt arguments

### `Did you forget to add @triton.jit ?`
- **Cause:** A `@tl.builtin` is being called from host code
- **Fix:** Switch to the host-side equivalent (e.g., `triton.tools.tensor_descriptor.TensorDescriptor.from_tensor`)

### `desc_ptr must be None or tlx.tensor_descriptor_ptr, got <class 'triton.language.core.constexpr'>`
- **Cause:** The constexpr unwrap shim is missing; arg defaulted to `None` got wrapped by the JIT
- **Fix:** Ensure `semantic_shims` patch is active (provides `tl._unwrap_if_constexpr`)

---

## References

- **Runner docs:** [`runner/CLAUDE.md`](runner/CLAUDE.md) — patch catalog, selection rules, diagnostic recipes
- **Main docs:** [`CLAUDE.md`](CLAUDE.md) — build instructions, known issues
- **uTLX build:** [`utlx/CLAUDE.md`](utlx/CLAUDE.md) — wheel contents, runtime setup
