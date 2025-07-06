#!/usr/bin/env python
"""Selects the optimal PyTorch wheel for the current HW and installs it."""
import subprocess, sys, platform

CUDA_WHL = (
    "https://download.pytorch.org/whl/cu124/torch-2.3.0%2Bcu124-cp311-cp311-linux_x86_64.whl"
)
CPU_WHL = (
    "https://download.pytorch.org/whl/cpu/torch-2.3.0-cp311-cp311-macosx_14_0_arm64.whl"
)

def main() -> None:
    system = platform.system()
    has_nvidia = False

    if system in ("Linux", "Windows"):
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            has_nvidia = True
        except Exception:
            pass

    wheel = CUDA_WHL if has_nvidia else CPU_WHL
    print(f"‣ Installing PyTorch from {wheel.split('/')[-1]}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])

if __name__ == "__main__":
    main()
