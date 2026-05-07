#!/usr/bin/env python3
"""
Install locally built wheels and run utlx core tests.

Usage:
    python test_runner.py                    # install + run core tests
    python test_runner.py --all              # run all tests (may segfault)
    python test_runner.py -k test_barriers   # pass extra args to pytest

Installs the triton and utlx wheels from dist/. If multiple triton wheels
exist, fails with an error — clean dist/ first.
"""

import glob
import os
import subprocess
import sys
import sysconfig


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
TRITON_EXT_REPO = os.environ.get("TRITON_EXT_REPO", os.path.expanduser("~/oss/triton-ext"))
UTLX_TEST_DIR = os.path.join(TRITON_EXT_REPO, "extensions", "utlx", "test")

CORE_TESTS = [
    "test_alloc_barriers.py",
    "test_barriers.py",
    "test_local_alloc.py",
    "test_mem_ops.py",
    "test_storage_alias.py",
    "test_types.py",
    "test_custom_stages.py",
]


def find_one_wheel(name):
    pattern = os.path.join(DIST_DIR, f"{name}-*.whl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"[test] ERROR: no {name} wheel in {DIST_DIR}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"[test] ERROR: multiple {name} wheels in {DIST_DIR}:", file=sys.stderr)
        for m in matches:
            print(f"  {os.path.basename(m)}", file=sys.stderr)
        print(f"Remove old wheels and keep one.", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def install_wheels():
    triton_whl = find_one_wheel("triton")
    utlx_whl = find_one_wheel("utlx")

    print(f"[test] Installing: {os.path.basename(triton_whl)}", file=sys.stderr)
    print(f"[test] Installing: {os.path.basename(utlx_whl)}", file=sys.stderr)
    result = subprocess.run(
        ["uv", "pip", "install", "--force-reinstall", triton_whl, utlx_whl],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[test] install failed: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    print("[test] Installed OK", file=sys.stderr)


def run_tests(test_files, pytest_args):
    if not os.path.isdir(UTLX_TEST_DIR):
        print(f"[test] ERROR: test dir not found: {UTLX_TEST_DIR}", file=sys.stderr)
        sys.exit(1)

    dist = sysconfig.get_paths()["purelib"]
    libutlx = os.path.join(dist, "utlx_plugin", "libutlx.so")
    if not os.path.isfile(libutlx):
        print(f"[test] ERROR: libutlx.so not found at {libutlx}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["TRITON_PLUGIN_PATHS"] = libutlx
    env.setdefault("TRITON_CACHE_PATH", os.path.expanduser("~/.triton"))

    targets = [os.path.join(UTLX_TEST_DIR, f) for f in test_files]
    cmd = [sys.executable, "-m", "pytest", "-v"] + pytest_args + targets
    print(f"[test] TRITON_PLUGIN_PATHS={libutlx}", file=sys.stderr)
    print(f"[test] Running {len(test_files)} test files", file=sys.stderr)

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    args = sys.argv[1:]
    run_all = "--all" in args
    pytest_args = [a for a in args if a != "--all"]

    install_wheels()

    if run_all:
        test_files = [f for f in os.listdir(UTLX_TEST_DIR)
                      if f.startswith("test_") and f.endswith(".py")]
    else:
        test_files = CORE_TESTS

    run_tests(sorted(test_files), pytest_args)
