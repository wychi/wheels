#!/usr/bin/env python3
"""
Run a uTLX kernel with plugin setup and monkey-patches.

Usage:
    python runner.py <kernel.py> [args...]
    python runner.py tiny_gemm.py

Assumes triton and utlx wheels are already installed.
"""

import builtins
import os
import sys
import sysconfig
from typing import Any, Optional, Tuple


def _setup_utlx():
    dist_packages = sysconfig.get_paths()["purelib"]
    libutlx_path = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
    if not os.path.isfile(libutlx_path):
        print(f"ERROR: libutlx.so not found at {libutlx_path}", file=sys.stderr)
        print("Install triton + utlx wheels first.", file=sys.stderr)
        sys.exit(1)
    os.environ["TRITON_PLUGIN_PATHS"] = libutlx_path

    if "triton" in sys.modules:
        print("[runner] WARNING: triton imported before uTLX setup, reloading libtriton", file=sys.stderr)
        import importlib
        importlib.reload(sys.modules["triton"]._C.libtriton)
    else:
        import triton

    print(f"[runner] Triton {triton.__version__}", file=sys.stderr)

    import utlx_plugin as tlx
    print(f"[runner] uTLX loaded: {tlx.__file__}", file=sys.stderr)


def _apply_tlx_patches():
    import triton.language as tl
    import triton.language.semantic as triton_semantic
    from triton import knobs

    def _prepare_legacy_load(self, ptr, mask, other, boundary_check, padding):
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError("`padding_option` or `boundary_check` argument is not supported for legacy loads")
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if other and other.type.is_block():
                raise ValueError("Other argument cannot be block type if pointer argument is not a block")
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
        self, lhs: tl.tensor, rhs: tl.tensor, acc: tl.tensor,
        input_precision: Optional[str], allow_tf32, max_num_imprecise_acc: int,
        out_dtype: tl.dtype, tlx_paired_ctas: bool = False,
    ) -> Tuple[Any]:
        input_precision = tl._unwrap_if_constexpr(input_precision)
        allow_tf32 = tl._unwrap_if_constexpr(allow_tf32)
        assert input_precision is None or tl._unwrap_if_constexpr(allow_tf32) is None
        if input_precision is None:
            supports_tf32 = "tf32" in self.builder.options.allowed_dot_input_precisions
            input_precision = knobs.language.fp32_default or ("tf32" if
                (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee")
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
        assert tl._unwrap_if_constexpr(lhs.shape[-1]) == tl._unwrap_if_constexpr(rhs.shape[-2])
        min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
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
            _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
            ret_scalar_ty = out_dtype
        M = lhs.type.shape[-2]
        N = rhs.type.shape[-1]
        K = lhs.type.shape[-1]
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
        if max_num_imprecise_acc is None:
            max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default if (lhs.dtype.is_fp8() and rhs.dtype.is_fp8()) else 0
        return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc, ret_ty)

    setattr(tl, "_unwrap_if_constexpr", _unwrap_if_constexpr)
    setattr(triton_semantic.TritonSemantic, "_prepare_legacy_load", _prepare_legacy_load)
    setattr(triton_semantic.TritonSemantic, "dot_precheck", dot_precheck)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [args...]", file=sys.stderr)
        sys.exit(1)

    kernel_file = sys.argv[1]
    if not os.path.isfile(kernel_file):
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(kernel_dir, kernel_file)
        if os.path.isfile(candidate):
            kernel_file = candidate
        else:
            print(f"ERROR: {kernel_file} not found", file=sys.stderr)
            sys.exit(1)

    _setup_utlx()
    _apply_tlx_patches()

    import runpy
    sys.argv = sys.argv[1:]
    runpy.run_path(kernel_file, run_name="__main__")


if __name__ == "__main__":
    main()
