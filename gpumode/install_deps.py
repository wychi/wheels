"""Install triton + utlx wheels at runtime. Pure function — URL resolution
lives in make_submission.py."""

import subprocess
import sys


def _install_custom_deps(triton_url, utlx_url):
    if "--no-install" in sys.argv:
        return

    print(f"[setup] Python: {sys.version}", file=sys.stderr)
    print(f"[setup] Installing triton from: {triton_url}", file=sys.stderr)
    print(f"[setup] Installing utlx from: {utlx_url}", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         triton_url, utlx_url],
        capture_output=True, text=True,
    )
    print(f"[setup] pip exit code: {result.returncode}", file=sys.stderr)
    print(f"[setup] pip stdout: {result.stdout[-500:]}", file=sys.stderr)
    if result.returncode != 0:
        print(f"[setup] pip stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
