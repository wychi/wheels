#!/usr/bin/env python3
"""Run a uTLX kernel with plugin setup and monkey-patches.

Usage:
    python runner/runner.py <kernel.py> [kernel_args...]

Patch selection is resolved by `tlx_patches.resolve_for_kernel`:
  1. The kernel module's `__tlx_patches__ = [...]` (or `"all"`) at top level.
  2. `runner/tlx_patches.toml` section matching the installed utlx commit.
  3. `runner/tlx_patches.toml` `[default]`.
  4. All patches registered with `default=True`.

Assumes triton and utlx wheels are already installed.
"""

import os
import sys
import sysconfig


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


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    kernel_file = args[0]
    if not os.path.isfile(kernel_file):
        # Fall back: look for the bare name under sibling kernels/ dir.
        kernels_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "kernels"))
        candidate = os.path.join(kernels_dir, kernel_file)
        if os.path.isfile(candidate):
            kernel_file = candidate
        else:
            print(f"ERROR: {kernel_file} not found", file=sys.stderr)
            sys.exit(1)

    _setup_utlx()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tlx_patches
    selected = tlx_patches.resolve_for_kernel(kernel_file)
    tlx_patches.apply(selected)

    import runpy
    sys.argv = args
    runpy.run_path(kernel_file, run_name="__main__")


if __name__ == "__main__":
    main()
