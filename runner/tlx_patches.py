"""Monkey patches that bridge `utlx_plugin` to the currently-installed Triton.

Why this file exists
====================
The pre-built `utlx_plugin` wheel was authored against an older Triton API.
Upstream Triton keeps moving (renamed bindings, tightened verifiers, refactored
code generator hooks). Rather than re-cutting wheels for every drift, we bridge
the gap in Python here. As uTLX matures and a patch becomes unnecessary, set
its `default=False` or delete it.

Usage
=====
    from tlx_patches import resolve_for_kernel, apply

    names = resolve_for_kernel("kernels/foo.py")  # consults config + kernel
    apply(names)

Each patch is registered with `@register("<name>", default=<bool>)` and is
applied in registration order (which respects dependencies; see notes below).

Selection sources, in priority order (see `resolve_for_kernel`):
  1. Kernel module declares `__tlx_patches__ = [...]` at top level. Read
     via AST — kernel is not executed for this lookup.
  2. `tlx_patches.toml` next to this file: section matching the installed
     utlx wheel commit (e.g. `[utlx."f3d635af"]`).
  3. `tlx_patches.toml` `[default]` section.
  4. All patches registered with `default=True`.

Note: avoid `from __future__ import annotations` here — `make_submission.py`
inlines this file into a bundled submission; future imports must be the first
statement in a file, and they wouldn't be after concatenation.
"""

import ast
import os
import sys
import tomllib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union


@dataclass
class _Patch:
    name: str
    fn: Callable[[], None]
    default: bool
    doc: str


PATCHES: list[_Patch] = []


def register(name: str, *, default: bool = True):
    """Register a patch function. Patches are applied in registration order."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        PATCHES.append(
            _Patch(name=name,
                   fn=fn,
                   default=default,
                   doc=(fn.__doc__ or "").strip()))
        return fn

    return decorator


def list_patches() -> list[tuple[str, bool, str]]:
    """Return [(name, default, first_doc_line)] for all registered patches."""
    return [(p.name, p.default, p.doc.splitlines()[0] if p.doc else "")
            for p in PATCHES]


def _all_default_names() -> list[str]:
    return [p.name for p in PATCHES if p.default]


def _validate_names(names: Iterable[str]) -> list[str]:
    by_name = {p.name: p for p in PATCHES}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown patch(es): {sorted(set(unknown))}")
    # Preserve registration order to respect inter-patch dependencies.
    selected = set(names)
    return [p.name for p in PATCHES if p.name in selected]


def apply(names: Iterable[str], *, verbose: bool = True) -> list[str]:
    """Apply patches by name. Returns the list applied (in registration order)."""
    by_name = {p.name: p for p in PATCHES}
    chosen = _validate_names(names)
    for name in chosen:
        if verbose:
            print(f"[tlx_patches] applying {name}", file=sys.stderr)
        by_name[name].fn()
    return chosen


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tlx_patches.toml")


def _utlx_commit() -> str | None:
    """Return the local-segment commit of the installed utlx wheel, or None."""
    try:
        import importlib.metadata as m
        version = m.version("utlx")
    except Exception:
        return None
    if "+git" in version:
        return version.split("+git", 1)[1]
    return None


def _read_kernel_decl(kernel_file: str) -> list[str] | None:
    """If the kernel's module top-level assigns `__tlx_patches__ = [...]`,
    return that list. Read via AST — the kernel is not executed."""
    try:
        with open(kernel_file, "r") as f:
            source = f.read()
    except OSError:
        return None
    try:
        tree = ast.parse(source, filename=kernel_file)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [
                t.id for t in node.targets if isinstance(t, ast.Name)
            ]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        if "__tlx_patches__" not in targets or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if literal == "all":
            return _all_default_names()
        if isinstance(literal, (list, tuple)):
            return [str(x) for x in literal]
        return None
    return None


def _expand(spec) -> list[str]:
    """Convert a TOML `patches` value to a concrete list of names."""
    if spec == "all":
        return _all_default_names()
    if isinstance(spec, list):
        return [str(x) for x in spec]
    raise ValueError(
        f"`patches` must be a list of names or the string 'all', got {spec!r}")


def resolve_for_kernel(kernel_file: str | None = None,
                       *,
                       config_file: str = CONFIG_FILE) -> list[str]:
    """Resolve the patch selection for a kernel run.

    Order: kernel decl > config commit match > config [default] > all defaults.
    """
    if kernel_file:
        decl = _read_kernel_decl(kernel_file)
        if decl is not None:
            return _validate_names(decl)

    if os.path.isfile(config_file):
        with open(config_file, "rb") as f:
            cfg = tomllib.load(f)
        commit = _utlx_commit()
        utlx_table = cfg.get("utlx") or {}
        if commit and commit in utlx_table:
            return _validate_names(_expand(utlx_table[commit].get("patches")))
        default_table = cfg.get("default")
        if default_table:
            return _validate_names(_expand(default_table.get("patches")))

    return _all_default_names()


# ---------------------------------------------------------------------------
# Patches
#
# Order matters where noted. Each patch states what it bridges, what upstream
# change made it necessary, and a hint for when it can be retired.
# ---------------------------------------------------------------------------


@register("semantic_shims")
def _semantic_shims() -> None:
    """Restore TritonSemantic methods uTLX expects: `_prepare_legacy_load`,
    `dot_precheck`, and `tl._unwrap_if_constexpr`.

    Upstream removed these from the public surface; uTLX still calls them in
    its mma/load paths.

    Retire when: uTLX's mma/load paths stop relying on these specific
    methods.
    """
    import builtins
    from typing import Any, Optional, Tuple

    import triton.language as tl
    import triton.language.semantic as triton_semantic
    from triton import knobs

    def _prepare_legacy_load(self, ptr, mask, other, boundary_check, padding):
        if not ptr.type.scalar.is_ptr():
            raise ValueError(
                f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError(
                "`padding_option` or `boundary_check` argument is not supported for legacy loads"
            )
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError(
                    "Mask argument cannot be block type if pointer argument is not a block"
                )
            if other and other.type.is_block():
                raise ValueError(
                    "Other argument cannot be block type if pointer argument is not a block"
                )
        if ptr.type.is_block():
            if mask is not None:
                ptr, mask = self.broadcast_impl_value(ptr, mask)
            if other is not None:
                ptr, other = self.broadcast_impl_value(ptr, other)
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)
        if other is not None:
            other = self.cast(other, elt_ty)
        if ptr.type.is_block():
            shape = ptr.type.get_block_shapes()
            dst_ty = tl.block_type(elt_ty, shape)
        else:
            dst_ty = elt_ty
        return dst_ty, ptr, mask, other, is_bool

    def _unwrap_if_constexpr(o):
        if isinstance(o, list):
            return [_unwrap_if_constexpr(x) for x in o]
        if isinstance(o, builtins.tuple):
            return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
        if isinstance(o, tuple):
            return tuple(_unwrap_if_constexpr(x) for x in o)
        return o.value if isinstance(o, tl.constexpr) else o

    def dot_precheck(
        self,
        lhs: tl.tensor,
        rhs: tl.tensor,
        acc: tl.tensor,
        input_precision: Optional[str],
        allow_tf32,
        max_num_imprecise_acc: int,
        out_dtype: tl.dtype,
        tlx_paired_ctas: bool = False,
    ) -> Tuple[Any]:
        input_precision = tl._unwrap_if_constexpr(input_precision)
        allow_tf32 = tl._unwrap_if_constexpr(allow_tf32)
        assert input_precision is None or tl._unwrap_if_constexpr(
            allow_tf32) is None
        if input_precision is None:
            supports_tf32 = "tf32" in self.builder.options.allowed_dot_input_precisions
            input_precision = knobs.language.fp32_default or (
                "tf32" if (supports_tf32 and
                           (allow_tf32 or allow_tf32 is None)) else "ieee")
        input_precision = tl._unwrap_if_constexpr(input_precision)
        out_dtype = tl._unwrap_if_constexpr(out_dtype)
        max_num_imprecise_acc = tl._unwrap_if_constexpr(max_num_imprecise_acc)
        acc = tl._unwrap_if_constexpr(acc)
        assert lhs.type.is_block() and rhs.type.is_block()
        if not (lhs.dtype.is_fp8() and rhs.dtype.is_fp8()):
            assert lhs.dtype == rhs.dtype
        if input_precision is None:
            input_precision = self.builder.options.default_dot_input_precision
        input_precision = self._str_to_dot_input_precision(input_precision)
        lhs_rank = len(lhs.shape)
        assert lhs_rank == len(rhs.shape)
        assert tl._unwrap_if_constexpr(
            lhs.shape[-1]) == tl._unwrap_if_constexpr(rhs.shape[-2])
        min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type,
                                                                rhs.type)
        assert (tl._unwrap_if_constexpr(lhs.shape[-2]) >= min_dot_size[0]
                and tl._unwrap_if_constexpr(lhs.shape[-1]) >= min_dot_size[2]
                and tl._unwrap_if_constexpr(rhs.shape[-1]) >= min_dot_size[1])
        if lhs.type.scalar.is_int():
            _0 = self.builder.get_int32(0)
            ret_scalar_ty = tl.int32
        elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
            _0 = self.builder.get_fp32(0)
            ret_scalar_ty = tl.float32
        else:
            _0 = self.builder.get_fp16(
                0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
            ret_scalar_ty = out_dtype
        M = lhs.type.shape[-2]
        N = rhs.type.shape[-1]
        K = lhs.type.shape[-1]
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder),
                                                   _0)
        else:
            acc_handle = acc.handle
        if max_num_imprecise_acc is None:
            max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default if (
                lhs.dtype.is_fp8() and rhs.dtype.is_fp8()) else 0
        return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc,
                ret_ty)

    setattr(tl, "_unwrap_if_constexpr", _unwrap_if_constexpr)
    setattr(triton_semantic.TritonSemantic, "_prepare_legacy_load",
            _prepare_legacy_load)
    setattr(triton_semantic.TritonSemantic, "dot_precheck", dot_precheck)


@register("dispatch_visit_with")
def _dispatch_visit_with() -> None:
    """Route `with tlx.async_tasks(): / with tlx.async_task(...):` in
    `@triton.jit` bodies to uTLX's custom code generator.

    Upstream's `CodeGenerator.visit_With` has no extension hook, so uTLX's
    `TLX_WITH_DISPATCH` table is never consulted. We patch `visit_With` to
    consult it before the default flow.

    Retire when: upstream gains a public hook for `with`-statement dispatch.

    Must run before `warp_specialize_codegen` (which mutates the dispatch
    table this patch consults).
    """
    import ast

    from triton.compiler import code_generator as cg
    from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH

    _orig = cg.CodeGenerator.visit_With

    def _patched(self, node):
        if len(node.items) == 1:
            ctx = node.items[0].context_expr
            if isinstance(ctx, ast.Call):
                try:
                    fn = self.visit(ctx.func)
                except Exception:
                    fn = None
                if fn is not None and fn in TLX_WITH_DISPATCH:
                    return TLX_WITH_DISPATCH[fn](self, node)
        return _orig(self, node)

    cg.CodeGenerator.visit_With = _patched


@register("make_tensor_descriptor", default=False)
def _make_tensor_descriptor() -> None:
    """Rewrite `tlx.make_tensor_descriptor` to emit a TMA descriptor whose
    result type carries an `NVMMASharedLayout` matching the destination
    shared memory.

    Retired in wheel commit `cba4ef9a`: `mem_ops.make_tensor_descriptor` now
    emits the gluon 5-arg binding with explicit `NVMMASharedLayout` directly.
    Kept here (not deleted) for older wheels (`f3d635af`, `47debefa`).

    Three problems being bridged:

    1. The JIT wraps a literal `None` arg into `constexpr(None)`; the
       original `is None` type check rejects that.
    2. uTLX's `mem_ops.make_tensor_descriptor` calls the regular 6-arg
       `ir.builder.create_make_tensor_descriptor` (block_shape + is_signed),
       which produces a `!tt.tensordesc<…>` with NO encoded layout
       attribute. Current Triton's `ttng.async_tma_copy_global_to_local`
       verifier rejects that with "TMA descriptor layout must match shared
       layout, but got descriptor layout <<NULL ATTRIBUTE>>".
    3. Gluon's 5-arg overload takes an explicit result type, so we build
       the tensordesc type with `get_tensor_descriptor_layout_type(
       block_type, is_signed, NVMMASharedLayout._to_ir())` and call gluon's
       binding directly. Layout selection mirrors
       `triton.experimental.gluon.language._layouts.NVMMASharedLayout
       .get_default_for(block_shape, dtype)` for the
       transposed=False / fp4_padded=False / no-CGA case, which is what
       `local_alloc` produces by default — keeping desc and dest layouts
       in sync.

    Retire when: uTLX's `make_tensor_descriptor` is updated for the new
    Triton signature, AND the JIT's constexpr wrapping is handled inside.

    Must run after `gluon_op_builder_swap` (relies on `_semantic.builder`
    being a `GluonOpBuilder` for `get_nvmma_shared_layout` /
    `get_tensor_descriptor_layout_type`).
    """
    import functools

    import triton.language.core as tl_core
    import utlx_plugin
    import utlx_plugin.mem_ops as _utlx_mem_ops
    from triton._C.libtriton import gluon_ir as _gluon_ir

    _gluon_create_mtd = _gluon_ir.GluonOpBuilder.create_make_tensor_descriptor

    def _default_nvmma_swizzle(block_shape, element_bitwidth):
        """Mirror NVMMASharedLayout.get_default_for swizzle selection
        (transposed=False, fp4_padded=False, no CGA)."""
        contig_bytes = block_shape[-1] * element_bitwidth // 8
        if contig_bytes >= 128 and contig_bytes % 128 == 0:
            swizzle = 128
        elif contig_bytes >= 64 and contig_bytes % 64 == 0:
            swizzle = 64
        elif contig_bytes >= 32 and contig_bytes % 32 == 0:
            swizzle = 32
        else:
            swizzle = 0
        flatten_outer = 1
        for s in block_shape[:-1]:
            flatten_outer *= s
        if len(block_shape) < 2 or flatten_outer < 8:
            swizzle = 0
        return swizzle

    @tl_core.builtin
    def _patched(desc_ptr=None,
                 base=None,
                 shape=None,
                 strides=None,
                 block_shape=None,
                 padding_option="zero",
                 _semantic=None,
                 **kwargs):
        desc_ptr = tl_core._unwrap_if_constexpr(desc_ptr)
        if desc_ptr is not None:
            raise NotImplementedError(
                "make_tensor_descriptor with explicit desc_ptr requires the "
                "7-arg create_make_tensor_descriptor binding which the "
                "current wheel doesn't expose. Pass desc_ptr=None and let "
                "the compiler auto-allocate via triton.set_allocator.")
        ndim = len(shape)
        assert 1 <= ndim <= 5
        assert len(strides) == ndim
        assert len(block_shape) == ndim
        shape_vals = [_semantic.make_scalar(x, tl_core.int32) for x in shape]
        strides_vals = [
            _semantic.make_scalar(tl_core._unwrap_if_constexpr(x),
                                  tl_core.int64) for x in strides
        ]
        block_shape = tl_core._unwrap_shape(block_shape)
        block_type = tl_core.block_type(base.type.element_ty, block_shape)
        is_signed_int = base.type.element_ty.is_int_signed()
        padding = _semantic._str_to_padding_option(padding_option)

        builder = _semantic.builder
        elt_ty = base.type.element_ty
        element_bitwidth = elt_ty.primitive_bitwidth
        rank = len(block_shape)
        swizzle = _default_nvmma_swizzle(block_shape, element_bitwidth)
        # iter34a (trimul bmm): per-call `transposed=True` for the fp32
        # C-buffer descriptor; default false keeps inputs unchanged.
        transposed = bool(
            tl_core._unwrap_if_constexpr(kwargs.pop("transposed", False)))
        layout_attr = builder.get_nvmma_shared_layout(swizzle,
                                                     element_bitwidth,
                                                     transposed, False, [], rank)
        result_ty = builder.get_tensor_descriptor_layout_type(
            block_type.to_ir(builder), is_signed_int, layout_attr)
        handle = _gluon_create_mtd(builder, result_ty, base.handle,
                                   [s.handle for s in shape_vals],
                                   [s.handle for s in strides_vals], padding)
        return tl_core.tensor_descriptor(handle, shape_vals, strides_vals,
                                         block_type)

    utlx_plugin.make_tensor_descriptor = _patched
    _utlx_mem_ops.make_tensor_descriptor = _patched


@register("wgmma_use_acc_default")
def _wgmma_use_acc_default() -> None:
    """Wrap `GluonOpBuilder.create_warpgroup_mma` to default `useAcc=None`
    to `get_int1(True)`.

    `tlx.async_dot` passes `None` for `useAcc`; the binding now requires a
    real `ir.value`.

    Retire when: uTLX's `mma_ops.async_dot` passes a proper `get_int1(...)`
    instead of `None`.
    """
    from triton._C.libtriton import gluon_ir as _gluon_ir

    _orig = _gluon_ir.GluonOpBuilder.create_warpgroup_mma

    def _patched(self, a, b, acc, use_acc, precision, max_num_imprecise,
                 is_async):
        if use_acc is None:
            use_acc = self.get_int1(True)
        return _orig(self, a, b, acc, use_acc, precision, max_num_imprecise,
                     is_async)

    _gluon_ir.GluonOpBuilder.create_warpgroup_mma = _patched


@register("broadcast_shape_overload")
def _broadcast_shape_overload() -> None:
    """Bridge `GluonOpBuilder.create_broadcast(value, shape_list)` calls
    coming out of `TritonSemantic` to the regular `ir.builder` overload.

    With the GluonOpBuilder swap active, pybind11 resolves
    `create_broadcast` to gluon's overload that takes an explicit `ir.type`.
    But Triton's `TritonSemantic.broadcast_impl_*` calls
    `builder.create_broadcast(handle, shape)` with a list of ints, matching
    the regular `ir.builder` form. We dispatch list-shape calls to the
    regular `ir.builder.create_broadcast` unbound method and pass `ir.type`
    calls through to gluon's native overload.

    Retire when: GluonOpBuilder.create_broadcast accepts a shape list, or
    uTLX moves to gluon natively.
    """
    from triton._C.libtriton import gluon_ir as _gluon_ir
    from triton._C.libtriton import ir as _ir

    _builder_create_broadcast = _ir.builder.create_broadcast
    _orig = _gluon_ir.GluonOpBuilder.create_broadcast

    def _patched(self, value, arg):
        if isinstance(arg, _ir.type):
            return _orig(self, value, arg)
        return _builder_create_broadcast(self, value, arg)

    _gluon_ir.GluonOpBuilder.create_broadcast = _patched


@register("gluon_op_builder_swap")
def _gluon_op_builder_swap() -> None:
    """Swap `CodeGenerator.builder` to `GluonOpBuilder` after `__init__`,
    keeping `TritonSemantic`.

    Most TLX-relevant ops (`create_warpgroup_mma`, `create_async_tma_*`,
    `create_local_alloc`, `create_memdesc_index`, `create_warp_specialize`,
    …) are bound only on `GluonOpBuilder`. Since `GluonOpBuilder` is a
    subclass of `ir.builder`, swapping it in keeps regular `tl.*` ops
    working while exposing the gluon-only `create_*` methods.

    Retire when: uTLX moves to gluon natively (sets `JITFunction.is_gluon`),
    or upstream binds these ops on the regular `TritonOpBuilder`.
    """
    from triton._C.libtriton import gluon_ir
    from triton.compiler import code_generator as cg

    _orig_init = cg.CodeGenerator.__init__

    def _target_name(options):
        """Mirror `<Backend>.get_target_name(options)`. Currently only CUDA
        (`cuda:NN` from `sm_NN` arch) is exercised."""
        arch = str(options.arch)
        if options.backend_name == "cuda" and arch.startswith("sm"):
            return f"cuda:{int(arch[2:])}"
        return f"{options.backend_name}:{arch}"

    def _patched_init(self,
                      context,
                      prototype,
                      gscope,
                      function_name,
                      jit_fn,
                      *,
                      options,
                      codegen_fns,
                      module_map,
                      is_gluon,
                      module=None,
                      is_kernel=False,
                      function_types=None,
                      noinline=False,
                      caller_context=None,
                      file_name=None,
                      begin_line=0,
                      begin_col=1):
        _orig_init(self,
                   context,
                   prototype,
                   gscope,
                   function_name,
                   jit_fn,
                   options=options,
                   codegen_fns=codegen_fns,
                   module_map=module_map,
                   is_gluon=is_gluon,
                   module=module,
                   is_kernel=is_kernel,
                   function_types=function_types,
                   noinline=noinline,
                   caller_context=caller_context,
                   file_name=file_name,
                   begin_line=begin_line,
                   begin_col=begin_col)
        if not is_gluon:
            new_builder = gluon_ir.GluonOpBuilder(context)
            new_builder.set_loc(file_name, begin_line, begin_col)
            new_builder.options = options
            new_builder.codegen_fns = codegen_fns
            new_builder.module_map = {} if module_map is None else module_map
            self.semantic.builder = new_builder
            self.builder = new_builder
            if module is None:
                self.module = new_builder.create_module()
                # Eagerly set ttg attributes so layout verifiers (which need
                # to resolve warps-per-CTA) succeed mid-codegen. Mirrors
                # `triton.experimental.gluon._runtime.GluonASTSource.make_ir`.
                self.module.set_attr(
                    "ttg.target",
                    new_builder.get_string_attr(_target_name(options)))
                self.module.set_attr(
                    "ttg.num-warps",
                    new_builder.get_int32_attr(options.num_warps))
                self.module.set_attr(
                    "ttg.num-ctas",
                    new_builder.get_int32_attr(options.num_ctas))
                self.module.set_attr(
                    "ttg.threads-per-warp",
                    new_builder.get_int32_attr(options.warp_size))
                if (options.backend_name == "cuda"
                        and getattr(options, "maxnreg", None) is not None):
                    self.module.set_attr(
                        "ttg.maxnreg",
                        new_builder.get_int32_attr(options.maxnreg))

    cg.CodeGenerator.__init__ = _patched_init


@register("async_load_native")
def _async_load_native() -> None:
    """Replace `tlx.async_load` / `async_load_commit_group` /
    `async_load_wait_group` with calls to GluonOpBuilder's native
    `create_async_copy_global_to_local` / `create_async_commit_group` /
    `create_async_wait_group`.

    The plugin's `utlx_async_load` op (C++ side, baked into `libutlx.so`)
    constructs `ttg.async_copy_global_to_local` with
    `operandSegmentSizes = array<i32: 0, 0, 0, 0>` regardless of how many
    operands it actually attaches. The verifier rejects this. Since current
    GluonOpBuilder exposes the upstream op directly, we bypass the plugin
    entirely for the non-bulk path.

    Async tokens become opaque sentinels (`async_token(None)`): commit/wait
    use the legacy CUDA cp.async group counter (no token plumbing), and
    `_flatten_ir_types` for `async_token_type` is already a no-op so a None
    handle round-trips harmlessly through liveins / region captures.

    Bulk (TMA) path is left untouched — it likely also needs migration but
    no kernel exercises it yet.

    Retire when: utlx's `mem_ops.async_load` is rewritten against gluon's
    `create_async_copy_global_to_local`.
    """
    import functools

    import triton.language as tl
    import triton.language.core as tl_core
    import utlx_plugin
    import utlx_plugin.mem_ops as _utlx_mem_ops
    from triton._C.libtriton import ir as _ir

    _orig_async_load = utlx_plugin.async_load

    @tl_core.builtin
    @functools.wraps(_orig_async_load)
    def _patched_async_load(src,
                            result,
                            mask=None,
                            other=None,
                            cache_modifier: str = "",
                            eviction_policy: str = "",
                            is_volatile: bool = False,
                            bulk: bool = False,
                            bulk_size=None,
                            barrier=None,
                            _semantic=None):
        bulk = tl._unwrap_if_constexpr(bulk)
        if bulk:
            return _orig_async_load(src,
                                    result,
                                    mask=mask,
                                    other=other,
                                    cache_modifier=cache_modifier,
                                    eviction_policy=eviction_policy,
                                    is_volatile=is_volatile,
                                    bulk=bulk,
                                    bulk_size=bulk_size,
                                    barrier=barrier,
                                    _semantic=_semantic)

        mask = tl._unwrap_if_constexpr(mask)
        other = tl._unwrap_if_constexpr(other)
        if mask is not None:
            mask = _semantic.to_tensor(mask)
        if other is not None:
            other = _semantic.to_tensor(other)

        if src.type.is_ptr() and src.type.element_ty.is_block():
            raise NotImplementedError(
                "async_load by block pointer is not supported yet")
        _, src, mask, other, _ = _semantic._prepare_legacy_load(
            src, mask, other, None, None)

        cache = _semantic._str_to_load_cache_modifier(cache_modifier)
        evict = _semantic._str_to_eviction_policy(eviction_policy)
        mask_handle = mask.handle if mask is not None else _ir.value()
        other_handle = other.handle if other is not None else _ir.value()
        # Native binding order: (smem_dest, ptr_src, mask, other, ...).
        _semantic.builder.create_async_copy_global_to_local(
            result.handle, src.handle, mask_handle, other_handle, cache,
            evict, bool(is_volatile))
        return utlx_plugin.async_token(None)

    @tl_core.builtin
    @functools.wraps(utlx_plugin.async_load_commit_group)
    def _patched_commit_group(tokens=None, _semantic=None):
        _semantic.builder.create_async_commit_group()
        return utlx_plugin.async_token(None)

    @tl_core.builtin
    @functools.wraps(utlx_plugin.async_load_wait_group)
    def _patched_wait_group(pendings, tokens=None, _semantic=None):
        pendings = tl._unwrap_if_constexpr(pendings)
        _semantic.builder.create_async_wait_group(int(pendings))
        return utlx_plugin.async_token(None)

    utlx_plugin.async_load = _patched_async_load
    utlx_plugin.async_load_commit_group = _patched_commit_group
    utlx_plugin.async_load_wait_group = _patched_wait_group
    _utlx_mem_ops.async_load = _patched_async_load
    _utlx_mem_ops.async_load_commit_group = _patched_commit_group
    _utlx_mem_ops.async_load_wait_group = _patched_wait_group


@register("wgmma_acc_layout_setup", default=False)
def _wgmma_acc_layout_setup() -> None:
    """Replace `tlx.async_dot`'s `utlx_require_nv_mma_layout(acc)` step with a
    direct splat of zero into a tensor with the correct `#mma` encoding.

    Retired in wheel commit `cba4ef9a`: `mma_ops.async_dot` now keeps the live
    acc value via the `utlx_require_nv_mma_layout` marker (no splat-zero), and
    a new C++ `TLXLayoutMarkerPattern` lowers the surviving
    `tlx.{require,release}_layout` ops to `ttg.convert_layout` (or folds
    same-encoding casts) so the `TritonGPURemoveLayoutConversions` /
    `TritonGPUReduceDataDuplication` walls disappear. Kept here for older
    wheels.

    The plugin's `tlx.require_layout` marker on the acc edge survives the
    `utlx_convert_triton_to_tritongpu` pass as an
    `unrealized_conversion_cast` from `#blocked` (added to the source
    `arith.constant` by the dialect conversion) to no-encoding (the marker's
    declared input type). The pass has no rewrite rule that absorbs this
    cast, so it remains live and the verifier rejects it. Bypassing the
    marker by emitting acc with the right encoding from the start avoids the
    materialization entirely.

    **Trade-off — single-shot acc only.** This patch always splats zero into
    the mma-typed acc, discarding the input acc value. That is correct for
    fresh-zero accumulators (`acc = tl.zeros(...); tlx.async_dot(a, b, acc)`)
    because `use_acc=True` then computes `0 + a*b = a*b`. Loop accumulators
    where `acc` carries forward across iterations would have their previous
    values silently dropped. Loop kernels need a different patch (or the
    wheel rebuild proper).

    Layout selection mirrors `triton.tools.triton_to_gluon_translator.
    hopper_helpers._mmav3_acc_layout`.

    Retire when: the wheel's `mma_ops.async_dot` builds the mma-encoded acc
    directly (e.g. via gluon's `warpgroup_mma_init` pattern), or the
    conversion pass learns to absorb `tlx.require_layout`.

    Must run after `gluon_op_builder_swap` (relies on `_semantic.builder`
    being a `GluonOpBuilder` for `get_mma_layout` / `get_distributed_ty`).
    """
    import functools
    import re

    import triton.language as tl
    import triton.language.core as tl_core
    import utlx_plugin
    import utlx_plugin.mma_ops as _mma_ops
    from utlx_plugin.mma_ops import (require_nv_mma_shared_layout,
                                     require_dot_operand_layout)

    def _cuda_capability(arch):
        m = re.fullmatch(r"sm(\d+)", str(arch))
        if not m:
            raise ValueError(f"unexpected arch {arch!r}")
        return int(m.group(1))

    # Hopper wgmma N-tile candidates, mirroring _mmav3_acc_layout.
    _VALID_N_FP = [
        256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152,
        144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24,
        16, 8
    ]
    _VALID_N_INT = [
        224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8
    ]

    def _mmav3_instr_and_warps(num_warps, c_shape, a_dtype):
        k = 256 // a_dtype.primitive_bitwidth
        valid_n = _VALID_N_FP if a_dtype.is_floating() else _VALID_N_INT
        m = 16
        m_warps = max(c_shape[0] // m, 1)
        n_warps_cap = max(num_warps // m_warps, 1)
        max_n = max(c_shape[1] // n_warps_cap, 8)
        instr_shape = None
        for n in valid_n:
            if c_shape[1] % n == 0 and n <= max_n:
                instr_shape = [m, n, k]
                break
        if instr_shape is None:
            raise RuntimeError(
                f"no valid wgmma instr shape for c_shape={c_shape}, "
                f"a_dtype={a_dtype}, num_warps={num_warps}")
        warps_per_tile = [4, 1]
        shape_per_warp = [16, instr_shape[1]]
        while True:
            if warps_per_tile[0] * warps_per_tile[1] >= num_warps:
                break
            if c_shape[0] > shape_per_warp[0] * warps_per_tile[0]:
                warps_per_tile[0] *= 2
            else:
                warps_per_tile[1] *= 2
        return instr_shape, warps_per_tile

    @tl_core.builtin
    @functools.wraps(utlx_plugin.async_dot)
    def _patched_async_dot(A,
                           B,
                           acc=None,
                           use_acc=None,
                           pred=None,
                           mBarriers=None,
                           two_ctas=False,
                           force_async=False,
                           input_precision=None,
                           out_dtype=tl.float32,
                           _semantic=None):
        if mBarriers is None:
            mBarriers = []
        (A, B, acc_handle, input_precision, max_num_imprecise_acc,
         ret_ty) = _semantic.dot_precheck(A, B, acc, input_precision, None,
                                          None, out_dtype, two_ctas)
        assert A.shape[0] >= 64, "M must be at least 64"
        assert A.shape[1] >= 16, "K must be at least 16"
        assert B.shape[1] >= 32, "N must be at least 32"

        capability = _cuda_capability(_semantic.builder.options.arch)
        if capability >= 100:
            # Blackwell tcgen05 path — fall through to original impl.
            return utlx_plugin.async_dot.__wrapped__(  # type: ignore[attr-defined]
                A,
                B,
                acc,
                use_acc=use_acc,
                pred=pred,
                mBarriers=mBarriers,
                two_ctas=two_ctas,
                force_async=force_async,
                input_precision=input_precision,
                out_dtype=out_dtype,
                _semantic=_semantic)

        # Hopper wgmma path.
        if isinstance(A, utlx_plugin.buffered_tensor) and \
                A.type.storage == utlx_plugin.storage_kind.smem:
            A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
        else:
            assert isinstance(A, tl.tensor)
            A_handle = A.handle
        B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

        builder = _semantic.builder
        num_warps = builder.options.num_warps
        c_shape = list(ret_ty.shape)
        a_dtype = A.dtype
        instr_shape, warps_per_tile = _mmav3_instr_and_warps(
            num_warps, c_shape, a_dtype)

        mma_layout = builder.get_mma_layout([3, 0], warps_per_tile, [],
                                            instr_shape)
        # acc element type from ret_ty.
        elt_ir_ty = ret_ty.scalar.to_ir(builder)
        mma_acc_ty = builder.get_distributed_ty(elt_ir_ty, c_shape, mma_layout)
        # Splat zero of acc element type into mma-encoded tensor.
        if ret_ty.scalar.is_fp32():
            zero = builder.get_fp32(0.0)
        elif ret_ty.scalar.is_fp16():
            zero = builder.get_fp16(0.0)
        elif ret_ty.scalar.is_int():
            zero = builder.get_int32(0)
        else:
            raise NotImplementedError(
                f"async_dot acc dtype {ret_ty.scalar} not handled")
        mma_acc_handle = builder.create_splat(mma_acc_ty, zero)

        if isinstance(A, tl.tensor):
            A_handle = require_dot_operand_layout(A, 0, mma_acc_handle, builder)

        # use_acc=True is correct: 0 + a*b == a*b. Lets us also accept any
        # incoming acc value harmlessly (it's discarded).
        output = builder.create_warpgroup_mma(A_handle, B_handle,
                                              mma_acc_handle, None,
                                              input_precision,
                                              max_num_imprecise_acc, True)
        # Strip the mma encoding via utlx_release_layout so downstream
        # python-level ops (`.to()`, `tl.store`) — which reconstruct types
        # from `ret_ty` (no encoding) — see consistent IR. Note: this still
        # leaves the IR with a downstream `ttg.convert_layout(no-enc →
        # blocked)` that the standard `tritongpu-remove-layout-conversions`
        # pass crashes on. Fixing that requires either a wheel rebuild
        # (proper conversion pattern for tlx.release_layout) or rewriting
        # the python-level type system to carry encodings through `.to()`.
        output = builder.utlx_release_layout([output])
        return tl.tensor(output, ret_ty)

    utlx_plugin.async_dot = _patched_async_dot
    _mma_ops.async_dot = _patched_async_dot


@register("warp_specialize_codegen")
def _warp_specialize_codegen() -> None:
    """Rewrite uTLX's `visit_withAsyncTasks` for the new `WarpSpecializeOp`
    IR shape.

    The legacy implementation calls `builder.create_warp_specialize_op(...)`
    plus `ws_op.get_partition_region` / `append_operand` for captures —
    these targeted a stale TritonOpBuilder extension that the prebuilt
    wheel doesn't expose. Current Triton's `WarpSpecializeOp` has only two
    regions: `defaultRegion` and `partitionOpHolder`, with the actual
    partition regions living inside a nested `WarpSpecializePartitionsOp`
    whose `explicitCaptures` operand carries the captures. We reimplement
    the codegen along the lines of `triton.experimental.gluon._semantic.
    warp_specialize`, using a `GluonOpBuilder` (sharing the `MLIRContext`)
    for the WS structural ops.

    Capture detection note: the original uTLX codegen used `self.used_vars`
    (a custom CodeGenerator attribute) to narrow captures to only-used
    outer vars. Current Triton's CodeGenerator doesn't expose that, so we
    over-capture from all non-constexpr `liveins`. Unused block args are
    harmless (DCE), but missing captures would violate IsolatedFromAbove
    on the partitions op.

    Retire when: uTLX's `code_generator.py:visit_withAsyncTasks` is
    rewritten upstream against the new IR shape.

    Must run after `dispatch_visit_with` (to refresh the dispatch table).
    """
    from triton._C.libtriton import gluon_ir
    import utlx_plugin.compiler.code_generator as _ucg
    from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH

    def _new_visit_withAsyncTasks(self, node):
        from triton.compiler.code_generator import (
            enter_sub_region,
            _is_list_like,
            _is_constexpr,
        )

        builder = self.builder
        gb = gluon_ir.GluonOpBuilder(self.context)

        def _sync_gb():
            gb.restore_insertion_point(builder.get_insertion_point())

        def _flatten_value_handles(val):
            handles = []
            if hasattr(val, "_flatten_ir"):
                val._flatten_ir(handles)
            else:
                handles.append(val.handle)
            return handles

        with enter_sub_region(self) as sr:
            liveins, _ = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            region_replica_id_stack = _ucg._get_region_replica_id_stack()

            stmts = node.body
            if not _is_list_like(stmts):
                stmts = [stmts]
            stmts = _ucg._resolve_async_task_stmts(self, stmts)

            has_non_default = False
            for stmt in stmts:
                task_check = _ucg._get_async_task(self, stmt)
                if not task_check.is_default:
                    has_non_default = True
                    break

            if not has_non_default:
                for stmt in stmts:
                    self.visit(stmt)
                return

            with _ucg.tlx_enter_sub_region():
                tmp_block = builder.create_block()
                builder.set_insertion_point_to_start(tmp_block)
                taskNumWarps: list[int] = []
                taskNumRegs: list[int] = []
                taskReplica: list[int] = []
                taskWarpGroupStartIds: list[int] = []
                perTaskNumWarps: list[int] = []
                perTaskStartIds: list[int] = []
                perTaskReplicates: list[int] = []
                region_replica_id_stack.append(-1)
                num_default = 0
                for stmt in stmts:
                    task = _ucg._get_async_task(self, stmt)
                    assert task.is_explict
                    assert task.replicate is not None
                    if task.is_default:
                        num_default += 1
                        if task.replicate > 1:
                            taskReplica.append(task.replicate - 1)
                            taskNumWarps.extend([builder.options.num_warps] *
                                                (task.replicate - 1))
                            if task.num_regs:
                                taskNumRegs.extend([task.num_regs] *
                                                   (task.replicate - 1))
                            if task.warp_group_start_id is not None:
                                taskWarpGroupStartIds.extend(
                                    [task.warp_group_start_id] *
                                    (task.replicate - 1))
                    else:
                        taskReplica.append(task.replicate)
                        taskNumWarps.extend([task.num_warps] * task.replicate)
                        if task.num_regs:
                            taskNumRegs.extend([task.num_regs] *
                                               task.replicate)
                        if task.warp_group_start_id is not None:
                            for r in range(task.replicate):
                                taskWarpGroupStartIds.append(
                                    task.warp_group_start_id +
                                    r * task.num_warps)
                            perTaskNumWarps.append(task.num_warps)
                            perTaskStartIds.append(task.warp_group_start_id)
                            perTaskReplicates.append(task.replicate)
                region_replica_id_stack.pop()

            assert num_default == 1, "Default task must be one and only one"
            tmp_block.erase()

            assert len(taskNumRegs) in [0, len(taskNumWarps)]
            assert len(taskWarpGroupStartIds) in [0, len(taskNumWarps)]
            if len(perTaskStartIds) > 0:
                _ucg._validate_warp_group_start_ids(
                    perTaskStartIds, perTaskNumWarps, perTaskReplicates,
                    builder.options.num_warps)

            # First pass: discover used vars by emitting partition bodies into
            # scratch blocks then erasing.
            self._set_insertion_point_and_loc(ip, last_loc)
            for stmt in stmts:
                task = _ucg._get_async_task(self, stmt)
                task_replicate = (task.replicate -
                                  1) if task.is_default else task.replicate
                if task_replicate > 0:
                    scratch = builder.create_block()
                    region_replica_id_stack.append(0)
                    builder.set_insertion_point_to_start(scratch)
                    with enter_sub_region(self):
                        self.visit(stmt)
                    region_replica_id_stack.pop()
                    scratch.erase()

            captures = sorted(name for name, val in liveins.items()
                              if not _is_constexpr(val))
            capture_handles = []
            for name in captures:
                val = liveins[name]
                if getattr(val, "__triton_aggregate__", False):
                    for field in val.type.fields:
                        v = getattr(val, field[0])
                        capture_handles.extend(_flatten_value_handles(v))
                else:
                    capture_handles.extend(_flatten_value_handles(val))
            arg_types = [h.get_type() for h in capture_handles]

            self._set_insertion_point_and_loc(ip, last_loc)
            _sync_gb()
            ws_op = gb.create_warp_specialize([], list(taskNumWarps))
            if len(taskNumRegs) > 0:
                ws_op.set_requested_registers(list(taskNumRegs))
            # warpGroupStartIds is not exposed via GluonOpBuilder; skip — not
            # used by current kernels. Add via op attr if needed.

            for stmt in stmts:
                task = _ucg._get_async_task(self, stmt)
                if not task.is_default:
                    continue
                region_replica_id_stack.append(0)
                default_block = builder.create_block_with_parent(
                    ws_op.get_default_region(), [])
                builder.set_insertion_point_to_start(default_block)
                with enter_sub_region(self):
                    self.visit(stmt)
                _sync_gb()
                gb.create_warp_yield([])
                region_replica_id_stack.pop()
                break

            holder_block = builder.create_block_with_parent(
                ws_op.get_partition_op_holder(), [])
            builder.set_insertion_point_to_start(holder_block)
            _sync_gb()
            partitions_op = gb.create_warp_specialize_partitions(
                capture_handles, sum(taskReplica))

            index = 0
            for stmt in stmts:
                task = _ucg._get_async_task(self, stmt)
                replicate_start = 1 if task.is_default else 0
                for i in range(replicate_start, task.replicate):
                    region_replica_id_stack.append(i)
                    partition_region = partitions_op.get_region(index)
                    index += 1
                    block = builder.create_block_with_parent(
                        partition_region, arg_types)
                    builder.set_insertion_point_to_start(block)
                    with enter_sub_region(self):
                        self.visit(stmt)
                    arg_idx = 0
                    for name in captures:
                        val = liveins[name]
                        if getattr(val, "__triton_aggregate__", False):
                            for field in val.type.fields:
                                v = getattr(val, field[0])
                                for h in _flatten_value_handles(v):
                                    arg = block.get_argument(arg_idx)
                                    arg_idx += 1
                                    block.replace_use_in_block_with(h, arg)
                        else:
                            for h in _flatten_value_handles(val):
                                arg = block.get_argument(arg_idx)
                                arg_idx += 1
                                block.replace_use_in_block_with(h, arg)
                    _sync_gb()
                    gb.create_warp_return()
                    region_replica_id_stack.pop()

            builder.set_insertion_point_after(ws_op.get_operation())

    _patched = _ucg.tlx_enter_sub_region()(_new_visit_withAsyncTasks)
    _ucg.visit_withAsyncTasks = _patched

    # Refresh dispatch table — it captured the old function reference at
    # import time.
    import triton.language.extra.tlx as _tlx_extra
    TLX_WITH_DISPATCH.clear()
    TLX_WITH_DISPATCH[_tlx_extra.async_tasks] = _patched
    TLX_WITH_DISPATCH[_tlx_extra.async_task] = _ucg.visit_withAsyncTask
    TLX_WITH_DISPATCH._initialized = True


@register("local_slice_fix")
def _local_slice_fix() -> None:
    """Fix `utlx_plugin.mem_ops.local_slice` for shared-memory buffers.

    The shipped wheel's `local_slice` calls
    `builder.create_memdesc_subslice(buffer.handle, offset, shape)` — a
    3-arg call matching an old binding. The current C++ binding is
    `(result_type: ir.type, source_value: ir.value, offsets: list)` —
    the result memdesc type comes FIRST. Calling without it raises
    "incompatible function arguments" at compile time and blocks any
    kernel that needs to slice an SMEM buffer along multiple dims
    (iter24's failed gate-LN epilogue, iter25's failed 2+2+1 fusion).

    The canonical pattern lives in `triton.experimental.gluon.language.
    _semantic.GluonSemantic.memdesc_slice`: build a new memdesc type with
    the slice shape (same dtype, num, storage, layout), get its IR via
    `to_ir(builder)`, then pass `(ty.to_ir, source.handle, offsets)`.
    `buffered_tensor_type.to_ir` already routes SMEM through
    `get_shared_mem_desc_ty`, so we reuse it.

    The TMEM branch of `local_slice` (which forwards to `subslice` ->
    `create_tmem_subslice`) is left untouched — different binding.

    Retire when: uTLX's `mem_ops.local_slice` is updated to construct
    the result memdesc type and pass it as the first arg to
    `create_memdesc_subslice`. (utlx-py — Python-only fix in
    `mem_ops.py`.)
    """
    from utlx_plugin import mem_ops as _mem_ops
    from utlx_plugin.types import buffered_tensor, buffered_tensor_type, storage_kind
    import triton.language.core as _tl_core

    @_tl_core.builtin
    def _local_slice_patched(buffer, offset, shape, _semantic=None):
        if buffer.type.storage == storage_kind.tmem:
            # Defer to original tmem handling.
            assert len(offset) == 2 and len(shape) == 2
            assert offset[0] == 0
            assert shape[0] == buffer.type.shape[0]
            return _mem_ops.subslice(buffer, offset[1], shape[1], _semantic=_semantic)

        # SMEM path — build the result memdesc type and call the binding
        # with `(type, value, offsets)`.
        slice_ty = buffered_tensor_type(
            buffer.type.element_ty,
            list(shape),
            buffer.type.num,
            buffer.type.storage,
            buffer.type.layout,
        )
        builder = _semantic.builder
        slice_handle = builder.create_memdesc_subslice(
            slice_ty.to_ir(builder), buffer.handle, offset
        )
        return buffered_tensor(
            slice_handle,
            buffer.type.element_ty,
            list(shape),
            buffer.type.num,
            buffer.type.storage,
            buffer.type.layout,
        )

    _mem_ops.local_slice = _local_slice_patched

    # tlx namespace re-exports the names; refresh the bound name.
    import utlx_plugin as _tlx
    _tlx.local_slice = _local_slice_patched


@register("nv_mma_shared_layout_to_ir_fix")
def _nv_mma_shared_layout_to_ir_fix() -> None:
    """Fix `nv_mma_shared_layout_encoding.to_ir` for the new GluonOpBuilder.

    The shipped wheel's `nv_mma_shared_layout_encoding.to_ir` calls
    `builder.make_nv_mma_shared_encoding_attr(shape, order, elem, ...)`
    — a stale builder method that doesn't exist on `GluonOpBuilder`.
    The current method is `get_nvmma_shared_layout(swizzle_bw,
    elem_bw, transposed, fp4_padded, cga_layout, rank)` (different
    signature, different name).

    This patch overrides `to_ir` to map utlx's per-dim CTA fields
    onto gluon's `cga_layout` representation and computes the swizzle
    byte-width from the layout's `shape`+`elemType`, mirroring
    `NVMMASharedLayout.get_default_for`. Surfaces when
    `local_slice` (or any code path that re-emits the layout type)
    walks through the layout's `to_ir`.

    Retire when: uTLX's `types.nv_mma_shared_layout_encoding.to_ir`
    is updated to call `get_nvmma_shared_layout` directly.
    """
    from utlx_plugin.types import nv_mma_shared_layout_encoding

    def _swizzle_byte_width(shape, element_bitwidth):
        """Same selection logic as NVMMASharedLayout.get_default_for."""
        contig_dim_bytes = shape[-1] * element_bitwidth // 8
        if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
            sw = 128
        elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
            sw = 64
        elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
            sw = 32
        else:
            sw = 0
        flatten_outer = 1
        for s in shape[:-1]:
            flatten_outer *= s
        if len(shape) < 2 or flatten_outer < 8:
            sw = 0
        return sw

    def _is_natural_order(order):
        """True when order is [rank-1, rank-2, ..., 0] (default, non-transposed)."""
        return list(order) == list(reversed(range(len(order))))

    def _patched_to_ir(self, builder):
        element_bitwidth = self.elemType.primitive_bitwidth
        rank = len(self.shape)
        swizzle_bw = (
            _swizzle_byte_width(self.shape, element_bitwidth)
            if self.swizzled else 0
        )
        transposed = not _is_natural_order(self.order)
        # utlx tracks numCTAsPerCGA/numCTASplit/numCTAOrder per dim. For the
        # default single-CTA case (all 1s), gluon's cga_layout is just []
        # ('no CGA tiling'). Non-default CGA layouts need explicit basis
        # vectors — fall back to empty for now and assert it's the default
        # case so we surface unsupported configurations rather than silently
        # mis-emit.
        is_default_cta = (
            list(self.numCTAsPerCGA) == [1] * rank
            and list(self.numCTASplit) == [1] * rank
        )
        if not is_default_cta:
            raise NotImplementedError(
                "nv_mma_shared_layout_to_ir_fix: non-default CTA layout "
                f"not yet mapped to gluon cga_layout (got per-CGA="
                f"{self.numCTAsPerCGA}, split={self.numCTASplit})."
            )
        cga_layout = []
        return builder.get_nvmma_shared_layout(
            swizzle_bw,
            element_bitwidth,
            transposed,
            bool(self.fp4Padded),
            cga_layout,
            rank,
        )

    nv_mma_shared_layout_encoding.to_ir = _patched_to_ir


@register("async_descriptor_load_eviction_policy", default=False)
def _async_descriptor_load_eviction_policy() -> None:
    """Plumb `eviction_policy` (and `cache_modifier`) through
    `tlx.async_descriptor_load` to the gluon TMA-load binding.

    The shipped wheel's `async_descriptor_load` accepts the kwarg and
    validates it but never forwards it to
    `create_async_tma_copy_global_to_local`. Pre-fix, the binding had
    no slot for it. After Triton rebuild that adds `cache=` and
    `evict=` parameters to the binding, this patch updates the wrapper
    to actually pass them.

    Requires: Triton wheel rebuilt with the gluon_ir.cc change that
    exposes `cache` and `evict` keyword args on
    `create_async_tma_copy_global_to_local`. Without that, the call
    raises 'incompatible function arguments'.

    Retire when: uTLX's `mem_ops.async_descriptor_load` is updated to
    pass the kwargs through, AND the binding accepts them.
    """
    from utlx_plugin import mem_ops as _mem_ops
    from utlx_plugin.mma_ops import require_nv_mma_shared_layout
    import triton.language.core as _tl_core
    import triton.language as _tl
    from triton._C.libtriton import ir as _ir

    _STR_TO_EVICT = {
        "": _ir.EVICTION_POLICY.NORMAL,
        "evict_first": _ir.EVICTION_POLICY.EVICT_FIRST,
        "evict_last": _ir.EVICTION_POLICY.EVICT_LAST,
    }
    _STR_TO_CACHE = {
        "": _ir.CACHE_MODIFIER.NONE,
        ".ca": _ir.CACHE_MODIFIER.CA,
        ".cg": _ir.CACHE_MODIFIER.CG,
        ".cs": _ir.CACHE_MODIFIER.CS,
        ".wb": _ir.CACHE_MODIFIER.WB,
        ".wt": _ir.CACHE_MODIFIER.WT,
    }

    @_tl_core.builtin
    def _patched(
        desc,
        result,
        offsets,
        barrier,
        pred=None,
        cache_modifier="",
        eviction_policy="",
        multicast_targets=None,
        _semantic=None,
    ):
        if multicast_targets is None:
            multicast_targets = []
        eviction_policy = _tl._unwrap_if_constexpr(eviction_policy)
        cache_modifier = _tl._unwrap_if_constexpr(cache_modifier)
        if eviction_policy not in _STR_TO_EVICT:
            raise ValueError(
                f"eviction_policy must be one of {list(_STR_TO_EVICT)}, "
                f"got '{eviction_policy}'"
            )
        if cache_modifier not in _STR_TO_CACHE:
            raise ValueError(
                f"cache_modifier must be one of {list(_STR_TO_CACHE)}, "
                f"got '{cache_modifier}'"
            )
        evict = _STR_TO_EVICT[eviction_policy]
        cache = _STR_TO_CACHE[cache_modifier]
        result_handle = require_nv_mma_shared_layout(result, True, _semantic.builder)
        offsets_ir = _semantic._convert_to_ir_values(offsets, require_i64=False)
        if pred is None:
            pred_handle = _semantic.builder.get_int1(True)
        else:
            pred_handle = pred.handle
        multicast = len(multicast_targets) > 0
        _semantic.builder.create_async_tma_copy_global_to_local(
            desc.handle, offsets_ir, barrier.handle, result_handle, pred_handle,
            multicast, None, cache, evict,
        )

    _mem_ops.async_descriptor_load = _patched
    import utlx_plugin as _tlx
    _tlx.async_descriptor_load = _patched
