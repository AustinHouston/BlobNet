from __future__ import annotations

import platform
import subprocess
import sys
import traceback

import torch


def main() -> int:
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.splitlines()[0]}")
    print(f"torch_version: {torch.__version__}")
    print(f"mac_ver: {platform.mac_ver()}")

    try:
        sw_vers = subprocess.check_output(["sw_vers"], text=True).strip()
    except Exception as exc:
        sw_vers = f"<failed to query sw_vers: {exc}>"
    print("sw_vers:")
    print(sw_vers)

    has_mps = hasattr(torch.backends, "mps")
    print(f"has_mps_backend: {has_mps}")
    if has_mps:
        print(f"mps_built: {torch.backends.mps.is_built()}")
        try:
            print(f"mps_available: {torch.backends.mps.is_available()}")
        except Exception as exc:
            print(f"mps_available_check_failed: {type(exc).__name__}: {exc}")

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")

    print("testing_mps_allocation: start")
    try:
        x = torch.randn((1024, 1024), device="mps")
        y = torch.randn((1024, 1024), device="mps")
        z = x @ y
        z_cpu = z[:2, :2].cpu()
        print("testing_mps_allocation: success")
        print(f"sample_result: {z_cpu}")
        return 0
    except Exception as exc:
        print(f"testing_mps_allocation: failed")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
