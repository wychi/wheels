# uTLX Enablement

This document tracks uTLX enablement work — bugs found, fixes applied, and patches needed for each utlx wheel commit. Each section is keyed by the utlx commit (the local segment of the wheel version).

## Commit Index

- [`f3d635af`](#f3d635af) — Initial wheel, requires full bridge layer
- [`1d7b7482`](#1d7b7482) — Built but never evaluated; superseded by `47debefa`
- [`47debefa`](#47debefa) — Same patch surface as `f3d635af`; `make_tensor_descriptor` extended to embed shared-layout into descriptor type for TMA loads (`hopper_ws.py`); `tlx.release_layout` wall and acc-loop trade-off remain
- [`cba4ef9a`](#cba4ef9a) — **Current wheel.** `tlx.release_layout` wall removed (C++ `TLXLayoutMarkerPattern` lowers markers to `ttg.convert_layout`); `mma_ops.async_dot` rewritten to preserve loop-carry acc; `mem_ops.make_tensor_descriptor` emits gluon binding with NVMMASharedLayout. Two patches retired (`make_tensor_descriptor`, `wgmma_acc_layout_setup`). `kernels/hopper_ws.py` and `kernels/tiny_gemm.py` both pass end-to-end. Open issues: TMA `eviction_policy` plumbing (resolved with local Triton patch), `tlx.local_slice` on contig-dim slices (partial), [`tl.split` segv on fp32 wgmma C-fragment tensors](#follow-up-tlsplit-segfaults-on-fp32-wgmma-derived-tensors-2026-05-11) (blocks S3-style `EPILOGUE_SUBTILE`).

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
**Status:** Skipped — superseded by `47debefa` before bisect was run.

---

## 47debefa

**Wheel:** `utlx-0.1.0+git47debefa-cp313-cp313-linux_x86_64.whl`  
**Triton:** 3.7.0+git7cff1f27  
**Status:** Tiny GEMM still blocked by `tlx.release_layout` lowering. Six patches required — none retired vs `f3d635af`.

### Bisect against `kernels/tiny_gemm.py`

Followed the [Patch re-evaluation playbook](#patch-re-evaluation-playbook-run-on-every-new-wheel) — start with `__tlx_patches__ = []`, add the patch addressing each successive failure until the next failure can't be bridged. Symptom-to-patch mapping is documented in [`runner/CLAUDE.md` → "Patch catalog"](runner/CLAUDE.md#patch-catalog).

Order in which patches were added (each addresses the failure left by the previous step):

1. `semantic_shims`
2. `gluon_op_builder_swap`
3. `broadcast_shape_overload`
4. `wgmma_use_acc_default`
5. `async_load_native`
6. `wgmma_acc_layout_setup` → **WALL** at `TritonGPURemoveLayoutConversions`

### Minimum required patches (this kernel)

```
semantic_shims
gluon_op_builder_swap
broadcast_shape_overload
wgmma_use_acc_default
async_load_native
wgmma_acc_layout_setup
```

### Not exercised by `tiny_gemm.py`

These patches did not appear in the bisect because the kernel doesn't use the affected APIs. They remain needed for other kernels:

- `dispatch_visit_with` — required by kernels using `with tlx.async_tasks():` (e.g. `kernels/hopper_ws.py`).
- `make_tensor_descriptor` — required by kernels using `tlx.make_tensor_descriptor` (e.g. `kernels/hopper_ws.py`).
- `warp_specialize_codegen` — paired with `dispatch_visit_with`; same trigger.

### Wheel-side delta vs `f3d635af`

**Zero patches retired.** The new wheel did not address any of the six failures surfaced by this bisect — same `_prepare_legacy_load` removal, same `create_warpgroup_mma` binding gap, same `create_broadcast` overload conflict, same `useAcc` argument rejection, same `utlx_async_load` `operandSegmentSizes` bug, same `tlx.require_layout` materialization wall.

### Outstanding blocker

The output-side `tlx.release_layout` marker survives both convert passes and leaves a malformed `ttg.convert_layout(no-encoding → blocked)` that `TritonGPURemoveLayoutConversions` crashes on. Fixing this requires either:

1. A C++ conversion pattern in `tlx-convert-triton-to-tritongpu` for `tlx.release_layout` (see [`utlx-cpp` table](#utlx-cpp--fix-in-next-wheel-rebuild-c)).
2. Dropping the markers entirely in `mma_ops.async_dot` and emitting `ttg.convert_layout` directly with the right encoded types (see [`utlx-py` table](#utlx-py--fix-in-next-wheel-rebuild-python-only) — this would need to extend `wgmma_acc_layout_setup`'s approach to the output side, plus python-level type plumbing through `.to()`).

Until either lands, `tiny_gemm.py` does not run end-to-end.

### Patch Configuration

```toml
[utlx."47debefa"]
patches = [
    "semantic_shims",
    "dispatch_visit_with",
    "make_tensor_descriptor",
    "wgmma_use_acc_default",
    "broadcast_shape_overload",
    "gluon_op_builder_swap",
    "async_load_native",
    "wgmma_acc_layout_setup",
    "warp_specialize_codegen",
]
```

> The 6 from the bisect plus the 3 not-exercised. `wgmma_acc_layout_setup` is included even though it doesn't fully unblock — without it the kernel fails one stage earlier, which is worse for diagnosis.

### Follow-up: `kernels/hopper_ws.py` enablement (2026-05-07)

Exercising the warp-specialized + TMA kernel surfaced one new patch-side
issue and confirmed the same wall as `tiny_gemm.py`.

**New finding — TMA descriptor missing layout encoding:**

With the previous `make_tensor_descriptor` patch (which routed to the
regular 6-arg `ir.builder.create_make_tensor_descriptor`), the result
type was `!tt.tensordesc<BMxBKxf16>` with no encoded layout. Current
Triton's `ttng.async_tma_copy_global_to_local` verifier rejects this:

```
'ttng.async_tma_copy_global_to_local' op TMA descriptor layout must
match shared layout, but got descriptor layout <<NULL ATTRIBUTE>> and
shared memory layout #ttg.nvmma_shared<{swizzlingByteWidth = 128,
transposed = false, elementBitWidth = 16}>
```

Updated `make_tensor_descriptor` patch now calls gluon's 5-arg overload
with an explicit result type built via
`get_tensor_descriptor_layout_type(block_type, is_signed,
NVMMASharedLayout._to_ir())`. Layout selection mirrors
`NVMMASharedLayout.get_default_for(block_shape, dtype)` for the
non-transposed / non-fp4 case — the same algorithm `local_alloc` uses by
default, so descriptor and destination layouts agree.

Same patch slot, same wheel-side classification (`utlx-py`); the docstring
in `runner/tlx_patches.py` and the catalog row in `runner/CLAUDE.md` have
been extended with the new symptom.

**Same wall as tiny_gemm + acc-loop trade-off:**

After the descriptor fix, the kernel reaches the documented
[outstanding blocker](#outstanding-blocker): `tlx.release_layout` on the
acc edge produces a `tensor<...xf32>` (no encoding), which feeds a
`ttg.convert_layout(no-encoding → blocked)` that the downstream pipeline
crashes on (`TritonGPUReduceDataDuplication` here vs.
`TritonGPURemoveLayoutConversions` for `tiny_gemm` — both downstream of
the same malformed cast). No new bridge attempted.

The IR also exposes the documented "single-shot acc only" trade-off in
`wgmma_acc_layout_setup`: `ttng.warp_group_dot` is fed a
`%cst (= dense<0.000000e+00>)` instead of the loop carry-in, so even past
the lowering wall the kernel would silently compute `a*b` for the last
iteration only. Loop accumulators in `hopper_ws.py` need either a wheel
rebuild (Python-level fix in `mma_ops.async_dot`) or a more sophisticated
patch.

**Status:** still blocked on the same `utlx-cpp` wall. Patch list
unchanged; `make_tensor_descriptor` reused with extended scope.

---

## cba4ef9a

**Wheel:** `utlx-0.1.0+gitcba4ef9a-cp313-cp313-linux_x86_64.whl`
**Triton:** 3.7.0+git7cff1f27
**Status:** Both `kernels/tiny_gemm.py` (single-shot) and
`kernels/hopper_ws.py` (warp-specialized + TMA + loop-carry acc) pass
end-to-end with `rel_err < 0.001`. Two patches retired vs `47debefa`.

### Wheel-side fixes

#### 1. C++ — `TLXLayoutMarkerPattern` (`utlx-cpp`)

Added an `OpConversionPattern<tlx::{Require,Release}LayoutOp>` to
`uTLXConversionPatterns.cpp:TLXConvertTritonToTritonGPU`. It runs during
the conversion pass and lowers each marker to `ttg.convert_layout`
between the source's encoding and the type-converter-assigned result
encoding. When the encodings happen to match (the result was
`tensor<...>` no-encoding and the converter assigned the same `#blocked`
that the source already has), the pattern forwards the operand directly
— a same-encoding no-op cast that earlier crashed
`TritonGPUReduceDataDuplication` is now folded away.

This single pattern fixes both the `tlx.release_layout` wall (output side
of `async_dot`) and any future `tlx.require_layout` calls that survive
typed-input bridging.

#### 2. Python — `mma_ops.async_dot` Hopper path (`utlx-py`)

Rewritten to preserve loop-carry acc:

- Wraps the live `acc_handle` with the existing `utlx_require_nv_mma_layout`
  marker (no splat-zero).
- Includes `create_warpgroup_mma_wait` inline (sync semantics; matches
  `triton.tools.triton_to_gluon_translator.hopper_helpers.tl_dot_mmav3`).
- On the output side, emits `ttg.convert_layout(mma → blocked)` directly
  (both encoded — valid), then `tlx.release_layout(blocked →
  no-encoding)` so the kernel's no-encoding iter_args type matches.
  After the conversion pass legalizes the no-encoding result to the same
  `#blocked`, the C++ pattern folds the marker.

`mma_ops.async_dot_wait` becomes a pass-through (the wait is now in
`async_dot`).

#### 3. Python — `mem_ops.make_tensor_descriptor` (`utlx-py`)

Switched from the regular 6-arg `ir.builder.create_make_tensor_descriptor`
(produces no-layout descriptor type) to the gluon 5-arg overload with an
explicit result type built via
`get_tensor_descriptor_layout_type(block_type, is_signed,
NVMMASharedLayout._to_ir())`. Layout selection mirrors
`NVMMASharedLayout.get_default_for(block_shape, dtype)` for the
non-transposed / non-fp4 case — matches what `local_alloc` produces by
default, so the TMA copy verifier accepts the descriptor. Also unwraps
the JIT's `constexpr(None)` for `desc_ptr`.

### Patch Configuration

```toml
[utlx."cba4ef9a"]
patches = [
    "semantic_shims",
    "dispatch_visit_with",
    "wgmma_use_acc_default",
    "broadcast_shape_overload",
    "gluon_op_builder_swap",
    "async_load_native",
    "warp_specialize_codegen",
]
```

Two patches retired (`make_tensor_descriptor`, `wgmma_acc_layout_setup`)
— marked `default=False` in `runner/tlx_patches.py` with `Retire when:`
notes pointing at this commit. The remaining 7 are still load-bearing.

### Patch obsoletion summary

| Patch                        | Status in `cba4ef9a` | Why                                                                 |
|------------------------------|----------------------|---------------------------------------------------------------------|
| `make_tensor_descriptor`     | Retired              | `mem_ops.make_tensor_descriptor` now embeds NVMMASharedLayout.      |
| `wgmma_acc_layout_setup`     | Retired              | `mma_ops.async_dot` preserves loop-carry; C++ pattern lowers markers. |
| `semantic_shims`             | Still required       | uTLX still calls removed `_prepare_legacy_load`/`dot_precheck`/`_unwrap_if_constexpr`. |
| `dispatch_visit_with`        | Still required       | Upstream still has no `visit_With` extension hook.                  |
| `wgmma_use_acc_default`      | Still required       | Other Hopper paths (`async_dot_scaled`, …) still pass `use_acc=None`. |
| `broadcast_shape_overload`   | Still required       | Coexists with `gluon_op_builder_swap`.                              |
| `gluon_op_builder_swap`      | Still required       | uTLX call sites still use `_semantic.builder` for gluon-only ops.   |
| `async_load_native`          | Still required       | Plugin's `utlx_async_load` op still has the `operandSegmentSizes` bug. |
| `warp_specialize_codegen`    | Still required       | `visit_withAsyncTasks` still uses the stale `WarpSpecializeOp` shape. |

### Follow-up: TMA `eviction_policy` plumbing (2026-05-08)

Investigated the chain that blocked PLAN_v4 iter21 (L2 cache-residency
hints on read-only TMA loads — see PROGRESS.md iter21). The wheel ships
`tlx.async_descriptor_load(..., eviction_policy='evict_last')` accepting
the kwarg but silently dropping it; uncovering this required four
distinct fixes spanning two repos:

1. **uTLX Python wrapper** (`mem_ops.async_descriptor_load`) — accepts
   `cache_modifier` / `eviction_policy` strings, validates them, never
   forwards. Bridged via runner-side patch
   `async_descriptor_load_eviction_policy` (`utlx-py`, default=False;
   becomes default=True only after the wrapper is fixed in-tree).
2. **Triton C++ binding** (`python/src/gluon_ir.cc`,
   `create_async_tma_copy_global_to_local`) — was a 7-arg signature
   with no slot for cache/evict. Extended to a 9-arg form with
   defaulted `cache=NONE` / `evict=NORMAL`. Backward-compatible.
3. **Triton MLIR op** (`TTNG_AsyncTMACopyGlobalToLocalOp`) — already
   carried `cache` and `evict` attributes via its `.td` definition;
   no change needed.
4. **Triton NVPTX lowering**
   (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`)
   — `AsyncTMACopyGlobalToLocalOpConversion` had an early-return
   `op.emitError("eviction policy not supported yet")`. Replaced with
   a call into the existing `createCachePolicy` helper that emits
   `createpolicy.fractional.L2::evict_<...>.b64` and threads the
   resulting policy register into the `cp.async.bulk.tensor` PTX as
   `.L2::cache_hint`. `computeCapability` plumbed through
   `populateLoadStoreOpToLLVMPatterns`.

**End-to-end status:** working. PTX dump on H100 shows
`createpolicy.fractional.L2::evict_last.b64 $cp, 1.0;` followed by
`cp.async.bulk.tensor.<rank>d.shared::cta.global.L2::cache_hint.mbarrier::complete_tx::bytes [...], $cp`.
The Triton-side patches live on local branch `wychi/tma-eviction-policy`
in `~/oss/triton` (commit `890498160`); they could land upstream as a
small backend addition.

**Performance verdict on the original target (trimul iter21):**
**~0% e2e gain on the trimul matmul kernel** (shape 6 control 5.253 ms
vs evict 5.273 ms, within bench noise). PLAN_v4's "+5-8% e2e" estimate
was wrong: the matmul's `B_g` weight (~480 KB) is small enough that
the round-robin scheduler already gets natural L2 reuse. The explicit
`evict_last` hint codifies what was already happening rather than
unlocking new behaviour. The iter21 lever is **DEAD as an optimization**
even with the wheel/upstream fixes; logged here to prevent re-investigation.

**Reclassification of the original ENABLEMENT.md `utlx-py` row for
`eviction_policy`:** the *runtime* fix is `utlx-py` (1 wrapper
function), but the underlying support requires `triton` (NVPTX backend
addition). This is a `triton` cluster issue that masquerades as a
`utlx-py` one. Future investigations of "uTLX wrapper accepts kwarg
but silently drops it" should check whether the upstream Triton
backend even implements the feature before committing to a uTLX patch.

### Follow-up: `tlx.local_slice` C++ binding fix (2026-05-08, partial)

While debugging the same iter21/iter24/iter25 surface, also investigated
`tlx.local_slice` on shared memory. The wheel's wrapper at
`utlx_plugin/mem_ops.py:355` calls
`builder.create_memdesc_subslice(buffer.handle, offset, shape)` — a
3-arg call matching an old binding. The current binding signature is
`(result_type, source_value, offsets)` with the result memdesc type
FIRST.

Bridged via two runner-side patches (both `default=False`):
- `local_slice_fix` — constructs the result `buffered_tensor_type` and
  calls `to_ir(builder)` for the result type, mirroring gluon's
  `memdesc_slice` reference impl.
- `nv_mma_shared_layout_to_ir_fix` — required because the layout's
  `to_ir` calls a stale `make_nv_mma_shared_encoding_attr` (also gone
  from `GluonOpBuilder`). Patches it to use `get_nvmma_shared_layout`
  with the correct 6-arg signature.

**Status:** patches load and reach the binding cleanly, but `local_slice`
on a buffer whose layout encodes the parent's shape (e.g., a slice on
the contiguous dim with a swizzle that no longer fits the slice) hits
the C++ verifier "block shape too small for swizzle byte size". A
correct fix needs the slice path to derive a smaller-swizzle layout
when slicing along the contiguous dim, or restrict slicing to outer
dims. Outer-dim slicing was confirmed to work past the layout layer
but ran into further tlx-API holes (`local_alloc(num=N)` adds a leading
dim that breaks naive `local_store` of a `[BM, BK]` source — the wheel
expects multi-buffer indexing patterns). Documented as TODO; no kernel
work currently depends on it.

### Follow-up: `tl.split` segfaults on fp32 wgmma-derived tensors (2026-05-11)

While porting S3's `EPILOGUE_SUBTILE` pattern into the trimul `bmm`
kernel (iter34b — split the [BLOCK_M_SPLIT, BN] = [64, 128] fp32 wgmma
accumulator into two BN/2 halves for two TMA stores), `tl.split` segvs
during AST→TTIR compilation.

**Symptom (faulthandler stack):**
```
Fatal Python error: Segmentation fault
  File ".../triton/language/semantic.py", line 692 in split
  File ".../triton/language/core.py", line 2142 in split
  File ".../triton/compiler/code_generator.py", line 1394 in call_Function
  ...
  File ".../triton/compiler/code_generator.py", line 959 in visit_If
  File ".../triton/compiler/code_generator.py", line 1094 in visit_While
```

The crash is in the C++ binding `builder.create_split(a.handle)`
(`semantic.py:692`). `_find_carries` is walking the trial `visit_If`
inside the consumer `while tile_id < num_tiles:` loop and trips on the
split call.

**Reproducer (minimal — toggle `EPILOGUE_SUBTILE=True` on the
existing S3 config and run any kernel that hits S3):**
```python
# matmul_kernel_tlx_ws (S3) consumer warpgroup, lines ~1753-1758
acc = tl.reshape(acc, (BLOCK_M_SPLIT, 2, BN // 2))
acc = tl.permute(acc, (0, 2, 1))
acc0, acc1 = tl.split(acc)        # ← segv here on fp32 wgmma acc
```

Confirmed **not iter34-specific.** Setting S3's existing untouched
`TLX_CONFIG["EPILOGUE_SUBTILE"] = True` (line 1782) — code in tree
since the iter15 era — crashes at the same line. The pattern was
authored but never exercised in production runs because every
`TLX_CONFIG` shipping default kept `EPILOGUE_SUBTILE = False`.

**Diagnosis:** `tl.split` requires the last dim of its input to equal
2; the C++ side then halves along that dim. The wgmma C-fragment for
fp32 has a per-lane register tiling that doesn't naturally map under
the `reshape(M, 2, N/2) → permute(M, N/2, 2) → split` rewrite.
Whatever layout-inference path `create_split` uses on a wgmma-rooted
tensor with a 2-element trailing dim hits a null deref or invalid
state. (S3 with bf16 wgmma C may also crash — not confirmed; both
S3 and bmm in our trimul build chose fp32 deliberately for cauchy
precision.)

**Workarounds (none of them clean enough to commit yet):**
- **Manual smem staging** — `tlx.local_alloc((BLOCK_M_SPLIT, BN), fp32)`,
  `tlx.local_store(slab, acc)`, then issue 2 TMA stores from
  `tlx.local_slice` halves. Bypasses `tl.split` entirely. Blocked by
  the [`local_slice` C++ binding issue](#follow-up-tlxlocal_slice-c-binding-fix-2026-05-08-partial)
  for slicing on the contiguous dim with the swizzle already fitted
  to the parent.
- **Two separate WGMMAs** — restructure the inner loop so each
  consumer WG runs two `[BLOCK_M_SPLIT, BN/2]` async_dots back to
  back. No `split` needed. Bigger restructure; ~half-day.
- **Skip subtile entirely** — go directly to register→TMA store
  (iter34c-style "skip the smem epilogue"). Avoids both `split` and
  `local_slice`.

**Fix classification:** `triton` (C++ layout-inference in
`create_split` for wgmma-rooted block tensors). The Python wrapper at
`semantic.py:685-696` is a thin pass-through; the bug is below that
in MLIR-land. Filing upstream is appropriate; the in-tree workaround
is to avoid `tl.split` on wgmma C-fragment tensors and use one of the
alternatives above.

**Net effect on iter34:** the planned-as-cheap iter34b lever
(EPILOGUE_SUBTILE port) is unreachable in its planned form on this
wheel. iter34a (transposed-swizzle flip on the C descriptor) shipped
its bmm-only -9.8% NCU duration win standalone; the next attempt at
the same goal will go via manual smem staging or direct register→TMA.

---

## Enablement Workflow

### When Cutting a New Wheel

1. **Build the wheel:**
   ```bash
   ./build_wheels.sh <triton-ext-commit>
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
| `make_tensor_descriptor`       | Update `mem_ops.make_tensor_descriptor` to use the current 5-arg gluon binding (`get_tensor_descriptor_layout_type(block_type, is_signed, NVMMASharedLayout._to_ir())`) so the result tensordesc carries an embedded shared-memory layout that matches the destination `local_alloc`. Without it, `ttng.async_tma_copy_global_to_local`'s verifier fires "TMA descriptor layout must match shared layout, but got descriptor layout <<NULL ATTRIBUTE>>". Also unwrap constexpr-`None` for `desc_ptr` inside. |
| `wgmma_use_acc_default`        | `mma_ops.async_dot` should pass `_semantic.builder.get_int1(True)` for `useAcc` instead of `None`. ~5-line fix. |
| `warp_specialize_codegen`      | Rewrite `compiler/code_generator.py:visit_withAsyncTasks` against the current `WarpSpecializeOp` IR shape (defaultRegion + partitionOpHolder + nested `WarpSpecializePartitionsOp` with `explicitCaptures`). |
| `async_load_native` (option a) | Drop the custom `utlx_async_load` op; have `mem_ops.async_load` call `create_async_copy_global_to_local` + `create_async_commit_group` + `create_async_wait_group` directly. |
| `async_descriptor_load_eviction_policy` | `mem_ops.async_descriptor_load` should pass `cache_modifier`/`eviction_policy` through to `create_async_tma_copy_global_to_local` instead of dropping them on the floor. ~10-line fix. **NOTE:** also requires the `triton` cluster fix below for the binding to even accept those kwargs and the lowering to emit them as PTX cache hints. Without that, this patch raises 'incompatible function arguments' at compile time. |
| `local_slice_fix` (partial)    | `mem_ops.local_slice` (SMEM branch) should construct the result memdesc type explicitly (mirror `triton.experimental.gluon.language._semantic.GluonSemantic.memdesc_slice`) and call `create_memdesc_subslice(result_type, source_handle, offsets)` — current call uses the old 3-arg form `(handle, offset, shape)` and crashes the binding. **Caveat:** also depends on `nv_mma_shared_layout_encoding.to_ir` being fixed (see `nv_mma_shared_layout_to_ir_fix`), and slicing along the contiguous dim of an NVMMA-swizzled buffer hits a verifier error — restrict to outer-dim slicing or derive a compatible swizzle for the slice. |
| `nv_mma_shared_layout_to_ir_fix` | `types.nv_mma_shared_layout_encoding.to_ir` calls a stale `make_nv_mma_shared_encoding_attr` that no longer exists on `GluonOpBuilder`. Replace with `get_nvmma_shared_layout(swizzle_byte_width, element_bitwidth, transposed, fp4_padded, cga_layout, rank)`, mirroring `NVMMASharedLayout._to_ir`. ~30-line fix including the swizzle-byte-width derivation from shape×elemtype. |

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
| TMA `eviction_policy` (and `cache_modifier`) | Two upstream changes prototyped on local branch `wychi/tma-eviction-policy` in `~/oss/triton` (commit `890498160`): (a) `python/src/gluon_ir.cc` — extend `create_async_tma_copy_global_to_local` pybind to take optional `cache=`/`evict=` kwargs; (b) `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp` — `AsyncTMACopyGlobalToLocalOpConversion` calls `createCachePolicy` (already in-tree) and emits `.L2::cache_hint` on `cp.async.bulk.tensor`. Verified end-to-end on H100; PTX dump shows `createpolicy.fractional.L2::evict_last.b64` + `cp.async.bulk.tensor.<rank>d.shared::cta.global.L2::cache_hint.mbarrier::complete_tx::bytes`. **Empirically: ~0% e2e gain on the trimul matmul** (B_g ~480 KB already enjoys natural L2 reuse from the round-robin scheduler) — kept in-tree as a wheel feature for cases where the working set actually thrashes L2. |

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

### `Fatal Python error: Segmentation fault` in `semantic.py:692 in split` / `create_split`
- **Cause:** `tl.split` C++ binding segvs on fp32 wgmma C-fragment tensors after `reshape→permute`. Affects every kernel that tries S3-style `EPILOGUE_SUBTILE` on a wgmma accumulator.
- **Fix:** Avoid `tl.split` on wgmma-rooted tensors. Either stage through smem and `tlx.local_slice`, or restructure to two smaller WGMMAs, or do the BN-half stores via direct register→TMA. Full writeup: [`cba4ef9a` → `tl.split` segfaults follow-up](#follow-up-tlsplit-segfaults-on-fp32-wgmma-derived-tensors-2026-05-11)

---

## References

- **Runner docs:** [`runner/CLAUDE.md`](runner/CLAUDE.md) — patch catalog, selection rules, diagnostic recipes
- **Main docs:** [`CLAUDE.md`](CLAUDE.md) — build instructions, known issues
- **uTLX build:** [`utlx/CLAUDE.md`](utlx/CLAUDE.md) — wheel contents, runtime setup
