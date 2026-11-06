#!/usr/bin/env python3
"""
NeuroSpark Serving Track Preparation & Inspection Harness
Author: Arunjyoti Sarma (IIT BHU Research Lab)
Target: Meta × Paradigm Open Kernel Challenge ($100k Prize Pool)
Architecture: NVIDIA Hopper H200 SXM (SM90a) × SGLang Serving Engine

This script automates preparation for the Serving Track (Opening Nov 10, 2026):
1. Verifies okc CLI installation and authentication.
2. Pulls the pinned SGLang serving codebase and patch rules via `okc serve init`.
3. Inspects the serving directory structure and patch guidelines.
4. Generates a template patch for integrating our 1.505x BF16 dense projection kernels.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

OKC_BIN = os.path.expanduser("~/.okc/bin/okc")
SERVING_DIR = Path("serving_workspace")
PATCH_FILE = Path("patch.diff")


def run_cmd(cmd_list, check=True):
    print(f"\n[*] Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(f"[stderr] {res.stderr.strip()}", file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res


def check_environment():
    print("=" * 70)
    print("  NEUROSPARK SERVING TRACK PREPARATION (META × PARADIGM OKC)")
    print("  IIT BHU Research Lab | Arunjyoti Sarma")
    print("=" * 70)

    # Check for okc CLI
    okc_path = shutil.which("okc") or (OKC_BIN if os.path.exists(OKC_BIN) else None)
    if not okc_path:
        print("[-] okc CLI not found in PATH or ~/.okc/bin/okc.")
        print("    If running in Colab/Linux, run: curl -fsSL https://openkernelchallenge.dev/install.sh | sh")
        return None

    print(f"[+] Found okc CLI: {okc_path}")
    # Verify auth
    res = subprocess.run([okc_path, "whoami"], capture_output=True, text=True)
    if res.returncode == 0 and "signed in as" in res.stdout:
        print(f"[+] Authenticated: {res.stdout.strip()}")
    else:
        print("[!] Not currently signed in. Run `okc login` first.")
    return okc_path


def init_serving_workspace(okc_path):
    print("\n" + "=" * 70)
    print("  STEP 1: INITIALIZE SGLANG SERVING WORKSPACE")
    print("=" * 70)

    if SERVING_DIR.exists():
        print(f"[*] Serving workspace '{SERVING_DIR}' already exists.")
    else:
        print(f"[*] Initializing serving workspace in '{SERVING_DIR}'...")
        run_cmd([okc_path, "serve", "init", str(SERVING_DIR)], check=False)

    # Check serving rules or files
    if SERVING_DIR.exists():
        print(f"[+] Serving directory contents:")
        for item in os.listdir(SERVING_DIR)[:15]:
            print(f"    - {item}")


def inspect_serving_status(okc_path):
    print("\n" + "=" * 70)
    print("  STEP 2: CHECK SERVING STATUS & LEADERBOARD")
    print("=" * 70)
    print("[*] Checking serving leaderboard...")
    run_cmd([okc_path, "serve", "leaderboard"], check=False)

    print("\n[*] Checking recent serving runs...")
    run_cmd([okc_path, "serve", "jobs"], check=False)


def generate_patch_template():
    print("\n" + "=" * 70)
    print("  STEP 3: PREPARE SERVING INTEGRATION PATCH TEMPLATE")
    print("=" * 70)
    template = """# NeuroSpark Serving Track Patch Template
# Incorporating Wave 8 Lossless 12-Bit BF16 Dense Projections & FlashDecode
# Target Hardware: 8x NVIDIA Hopper H200 SXM (SM90a)

# Instructions:
# 1. Modify the SGLang model runner inside serving_workspace/
# 2. Run: okc serve diff serving_workspace
# 3. Test: okc serve check patch.diff
# 4. Submit (Opening Nov 10): okc serve submit patch.diff
"""
    with open("serving_patch_guide.txt", "w", encoding="utf-8") as f:
        f.write(template)
    print("[+] Created 'serving_patch_guide.txt'.")


def main():
    okc = check_environment()
    if okc:
        init_serving_workspace(okc)
        inspect_serving_status(okc)
        generate_patch_template()
    else:
        print("\n[!] Please run this script in Google Colab or on a Linux GPU machine with okc installed.")


if __name__ == "__main__":
    main()
