"""Install triton + utlx wheels at runtime (used by make_submission.py)."""

import subprocess
import sys

TRITON_WHEEL_URL = "https://github.com/wychi/wheels/releases/download/triton-3.7.0-be8855ac/triton-3.7.0+gitbe8855ac-cp313-cp313-linux_x86_64.whl"
UTLX_WHEEL_URL = "https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl"


def _install_custom_deps():
    if "--no-install" in sys.argv:
        return

    print(f"[setup] Python: {sys.version}", file=sys.stderr)
    print(f"[setup] Installing triton from: {TRITON_WHEEL_URL}", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         f"triton @ {TRITON_WHEEL_URL}", f"utlx @ {UTLX_WHEEL_URL}"],
        capture_output=True, text=True,
    )
    print(f"[setup] pip exit code: {result.returncode}", file=sys.stderr)
    print(f"[setup] pip stdout: {result.stdout[-500:]}", file=sys.stderr)
    if result.returncode != 0:
        print(f"[setup] pip stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
