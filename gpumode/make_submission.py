#!/usr/bin/env python3
"""
Generate a self-contained submission with uTLX setup.

Reads install_deps.py and runner.py, trims header docstrings and
main()/if-__name__ blocks, then appends the submission file verbatim.

Usage:
    python make_submission.py submission.py              # prints to stdout
    python make_submission.py submission.py -o out.py    # writes to file

Wheel URL resolution order:
    1. releases.json [0] (latest entry — gh_release.sh prepends)
    2. DEFAULT_TRITON_URL / DEFAULT_UTLX_URL (last-known-good)
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_DIR = os.path.join(SCRIPT_DIR, "..", "runner")
RUNNER_PATH = os.path.join(RUNNER_DIR, "runner.py")
PATCHES_PATH = os.path.join(RUNNER_DIR, "tlx_patches.py")
INSTALL_PATH = os.path.join(SCRIPT_DIR, "install_deps.py")
RELEASES_JSON = os.path.join(SCRIPT_DIR, "..", "releases.json")


def read_source(path):
    with open(path) as f:
        return f.read()


def strip_header(source):
    """Remove shebang and module docstring."""
    lines = source.split("\n")
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and lines[i].startswith('"""'):
        if lines[i].count('"""') >= 2:
            i += 1
        else:
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


def strip_main(source):
    """Remove def main() and if __name__ block (assumed to be at the end)."""
    return re.sub(
        r"\n*^def main\(\):.+",
        "",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )


def _resolve_urls():
    """Resolution order: releases.json [0] > DEFAULT_*."""
    try:
        with open(RELEASES_JSON) as f:
            releases = json.load(f)
    except FileNotFoundError:
        releases = None
    if releases:
        return releases[0]["triton"], releases[0]["utlx"]

    # Last-known-good wheel URLs — used when releases.json is missing or empty.
    DEFAULT_TRITON_URL = "https://github.com/wychi/wheels/releases/download/triton-3.7.0-be8855ac/triton-3.7.0+gitbe8855ac-cp313-cp313-linux_x86_64.whl"
    DEFAULT_UTLX_URL = "https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl"
    return DEFAULT_TRITON_URL, DEFAULT_UTLX_URL


def _resolve_patches(input_path):
    """Resolve patch list at generation time. Tries the kernel's
    `__tlx_patches__` decl first, otherwise applies all defaults — the
    runtime container installs a wheel whose commit may not match the
    [utlx.\"<commit>\"] entries in tlx_patches.toml, so don't rely on
    runtime TOML lookup.
    """
    sys.path.insert(0, RUNNER_DIR)
    try:
        import tlx_patches
    finally:
        sys.path.pop(0)
    decl = tlx_patches._read_kernel_decl(input_path)
    return decl if decl is not None else tlx_patches._all_default_names()


def generate(input_path, triton_url, utlx_url):
    install = strip_header(read_source(INSTALL_PATH)).rstrip("\n")
    runner = strip_main(strip_header(read_source(RUNNER_PATH))).rstrip("\n")
    patches_module = strip_header(read_source(PATCHES_PATH)).rstrip("\n")
    kernel = read_source(input_path).rstrip("\n")
    name = os.path.basename(input_path)
    selected = _resolve_patches(input_path)

    return f"""\
#!/usr/bin/env python3
\"\"\"Auto-generated submission with uTLX setup.
Do not edit — regenerate with: make_submission.py {name}
\"\"\"

# --- Wheel install (from install_deps.py) ---

{install}


# --- uTLX setup (from runner.py) ---

{runner}


# --- Patch registry (from tlx_patches.py) ---

{patches_module}


_install_custom_deps({triton_url!r}, {utlx_url!r})
_setup_utlx()
apply({selected!r})


# --- Kernel (from {name}) ---

{kernel}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate self-contained uTLX submission")
    parser.add_argument("input", help="Kernel submission file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    for path, label in [(args.input, "input"), (RUNNER_PATH, "runner.py"), (PATCHES_PATH, "tlx_patches.py"), (INSTALL_PATH, "install_deps.py")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    triton_url, utlx_url = _resolve_urls()
    result = generate(args.input, triton_url, utlx_url)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        os.chmod(args.output, 0o755)
        print(f"Generated: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
