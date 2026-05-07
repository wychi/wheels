#!/usr/bin/env bash
# make_submission.sh — Wrap a clean submission.py into a self-contained
# submission_tlx.py with wheel install, plugin setup, and monkey-patches.
#
# Usage:
#   ./make_submission.sh submission.py              # prints to stdout
#   ./make_submission.sh submission.py -o out.py    # writes to file
#
# The input submission.py should:
#   - Follow the gpumode template (define custom_kernel(data))
#   - Import utlx_plugin as tlx
#   - NOT contain install/setup/monkey-patch code
#
# Environment:
#   TRITON_WHEEL_URL  — override Triton wheel URL
#   UTLX_WHEEL_URL    — override uTLX wheel URL

set -euo pipefail

INPUT="${1:?Usage: $0 <submission.py> [-o output.py]}"
OUTPUT=""

if [ "${2:-}" = "-o" ]; then
    OUTPUT="${3:?-o requires a filename}"
fi

[ -f "$INPUT" ] || { echo "ERROR: $INPUT not found" >&2; exit 1; }

TRITON_WHEEL_URL="${TRITON_WHEEL_URL:-https://github.com/wychi/wheels/releases/download/triton-3.7.0-be8855ac/triton-3.7.0+gitbe8855ac-cp313-cp313-linux_x86_64.whl}"
UTLX_WHEEL_URL="${UTLX_WHEEL_URL:-https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl}"

generate() {
cat << 'PREAMBLE_START'
#!/usr/bin/env python3
"""
Auto-generated submission with uTLX setup.
Do not edit — regenerate with: ./make_submission.sh submission.py -o submission_tlx.py
"""

import builtins
import os
import subprocess
import sys
import sysconfig
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# uTLX Setup (auto-generated)
# ---------------------------------------------------------------------------

def _install_custom_deps():
    if "--no-install" in sys.argv:
        return

PREAMBLE_START

cat << PREAMBLE_URLS
    TRITON_WHEEL_URL = "${TRITON_WHEEL_URL}"
    UTLX_WHEEL_URL = "${UTLX_WHEEL_URL}"
PREAMBLE_URLS

cat << 'PREAMBLE_INSTALL'

    print(f"[DEBUG] Python: {sys.version}", file=sys.stderr)
    print(f"[DEBUG] Installing triton from: {TRITON_WHEEL_URL}", file=sys.stderr)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                             f"triton @ {TRITON_WHEEL_URL}",
                             f"utlx @ {UTLX_WHEEL_URL}"],
                            capture_output=True, text=True)
    print(f"[DEBUG] pip exit code: {result.returncode}", file=sys.stderr)
    print(f"[DEBUG] pip stdout: {result.stdout[-500:]}", file=sys.stderr)
    if result.returncode != 0:
        print(f"[DEBUG] pip stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


def _setup_utlx():
    dist_packages = sysconfig.get_paths()["purelib"]
    libutlx_path = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
    assert os.path.isfile(libutlx_path), f"libutlx.so not found at {libutlx_path}"
    os.environ["TRITON_PLUGIN_PATHS"] = libutlx_path

    # Reload libtriton so it picks up the plugin
    import triton
    print(f"[DEBUG] Triton: {triton.__version__}", file=sys.stderr)

    import importlib
    importlib.reload(triton._C.libtriton)
    print(f"[DEBUG] uTLX loaded: {libutlx_path}", file=sys.stderr)


_install_custom_deps()
_setup_utlx()


# ---------------------------------------------------------------------------
# Monkey-patches (auto-generated)
# ---------------------------------------------------------------------------

import triton
import triton.language as tl
import triton.language.semantic as triton_semantic
from triton import knobs


def _apply_tlx_patches():

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

    setattr(triton.language, "_unwrap_if_constexpr", _unwrap_if_constexpr)
    setattr(triton_semantic.TritonSemantic, "_prepare_legacy_load", _prepare_legacy_load)
    setattr(triton_semantic.TritonSemantic, "dot_precheck", dot_precheck)


_apply_tlx_patches()

PREAMBLE_INSTALL

# Strip imports that the preamble already provides, then emit the user's kernel code.
# We keep all lines except redundant imports and the original shebang/docstring header.
python3 -c "
import sys

skip_imports = {
    'import builtins',
    'import os',
    'import subprocess',
    'import sys',
    'import sysconfig',
    'from typing import',
    'import triton',
    'import triton.language as tl',
    'import triton.language.semantic',
    'from triton import knobs',
}

lines = open(sys.argv[1]).readlines()
in_docstring = False
skipped_header = False

print()
print('# ---------------------------------------------------------------------------')
print('# Kernel (from ${INPUT})')
print('# ---------------------------------------------------------------------------')
print()

for line in lines:
    stripped = line.rstrip()

    # Skip shebang
    if stripped.startswith('#!') and not skipped_header:
        skipped_header = True
        continue

    # Skip module docstring
    if stripped.startswith('\"\"\"') and not in_docstring:
        in_docstring = True
        if stripped.count('\"\"\"') >= 2:
            in_docstring = False
        continue
    if in_docstring:
        if '\"\"\"' in stripped:
            in_docstring = False
        continue

    # Skip redundant imports
    if any(stripped.startswith(s) for s in skip_imports):
        continue

    print(line, end='')
" "$INPUT"
}

if [ -n "$OUTPUT" ]; then
    generate > "$OUTPUT"
    chmod +x "$OUTPUT"
    echo "Generated: $OUTPUT" >&2
else
    generate
fi
