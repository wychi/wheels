#!/usr/bin/env python3
"""
Generate a self-contained submission with uTLX setup.

Reads install_deps.py and runner.py, trims header docstrings and
main()/if-__name__ blocks, then appends the submission file verbatim.

Usage:
    python make_submission.py submission.py              # prints to stdout
    python make_submission.py submission.py -o out.py    # writes to file

To change wheel URLs, edit install_deps.py.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(SCRIPT_DIR, "..", "kernels", "runner.py")
INSTALL_PATH = os.path.join(SCRIPT_DIR, "install_deps.py")


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


def generate(input_path):
    install = strip_header(read_source(INSTALL_PATH)).rstrip("\n")
    runner = strip_main(strip_header(read_source(RUNNER_PATH))).rstrip("\n")
    kernel = read_source(input_path).rstrip("\n")
    name = os.path.basename(input_path)

    return f"""\
#!/usr/bin/env python3
\"\"\"Auto-generated submission with uTLX setup.
Do not edit — regenerate with: make_submission.py {name}
\"\"\"

# --- Wheel install (from install_deps.py) ---

{install}


# --- uTLX setup + patches (from runner.py) ---

{runner}


_install_custom_deps()
_setup_utlx()
_apply_tlx_patches()


# --- Kernel (from {name}) ---

{kernel}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate self-contained uTLX submission")
    parser.add_argument("input", help="Kernel submission file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    for path, label in [(args.input, "input"), (RUNNER_PATH, "runner.py"), (INSTALL_PATH, "install_deps.py")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    result = generate(args.input)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        os.chmod(args.output, 0o755)
        print(f"Generated: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
