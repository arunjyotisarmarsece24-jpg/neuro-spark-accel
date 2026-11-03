import json
import os

all_in_one_code = '''# ==============================================================================
# 🚀 ALL-IN-ONE OKC INSTALL, AUTH, DEPLOY & SUBMIT (MUSE_BF16_PROJ)
# ==============================================================================
import os
import pty
import select
import subprocess
import time

# 1. Install okc CLI if needed and set PATH
if not os.path.exists("/root/.okc/bin/okc"):
    print("[1/5] Installing okc CLI...")
    subprocess.run("curl -fsSL https://openkernelchallenge.dev/install.sh | sh", shell=True, check=True)

os.environ["PATH"] = "/root/.okc/bin:" + os.environ["PATH"]

# 2. Check if already signed in, otherwise authenticate via Pseudo-Terminal
print("\\n[2/5] Checking authentication...")
check = subprocess.run(["/root/.okc/bin/okc", "problems"], capture_output=True, text=True)

needs_login = ("not signed in" in check.stdout.lower() or "not signed in" in check.stderr.lower() or check.returncode != 0)

if needs_login:
    print("[*] Starting GitHub Device Flow Login...")
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        ["/root/.okc/bin/okc", "login"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True
    )
    os.close(slave)

    # Wait 1.0 second and send 'y\\n' through the real PTY
    time.sleep(1.0)
    os.write(master, b"y\\n")

    # Read output and display device code
    while True:
        r, _, _ = select.select([master], [], [], 2.0)
        if r:
            try:
                data = os.read(master, 1024)
                if not data:
                    break
                print(data.decode("utf-8", errors="ignore"), end="", flush=True)
            except OSError:
                break
        elif proc.poll() is not None:
            break

    proc.wait()
    print("\\n✓ Authentication completed!")
else:
    print("✓ Already authenticated!")

# 3. Initialize muse_bf16_proj workspace
print("\\n[3/5] Initializing muse_bf16_proj workspace...")
subprocess.run(["/root/.okc/bin/okc", "init", "muse_bf16_proj"])

# 4. Deploy our optimized Hopper H200 kernel into muse_bf16_proj/submission.py
print("\\n[4/5] Deploying optimized Hopper H200 kernel to muse_bf16_proj/submission.py...")
optimized_code = """# okc: problem=muse_bf16_proj
\\\"\\\"\\\"Optimized Dense Projections for Muse Spark Open on Hopper H200.
Author: Arunjyoti Sarma (IIT BHU Research Lab)

Key Techniques:
1. Zero-Allocation Hot Path: Pre-slice and pre-contiguous all weight buffers during untimed prepare().
2. Pre-Transposed Layouts: Eliminate runtime matrix transpositions.
3. Dual-Path Router: Pre-transposed FP32 buffer for tokens <= 8, split BF16 GEMM for tokens > 8.
4. Native cuBLASLt Dispatch with Zero Slicing Overhead.
\\\"\\\"\\\"

from typing import TypedDict
import torch
import torch.nn.functional as F

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    HAS_SGLANG = True
except Exception:
    HAS_SGLANG = False
    _bf16_gemm_dispatch_impl = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights: Weights):
    state = {}
    for name, w in weights.items():
        w_contig = w.contiguous()
        state[name] = w_contig
        state[f"{name}_t"] = w_contig.t().contiguous()

    unpooler = weights["unpooler"]
    state["unpooler_1024"] = unpooler[:1024].contiguous()
    state["unpooler_1024_t"] = unpooler[:1024].t().contiguous()
    state["unpooler_8192"] = unpooler.contiguous()
    state["unpooler_8192_t"] = unpooler.t().contiguous()

    router = weights["router"]
    state["router_t"] = router.t().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer = cfg["layer"]

    if layer == "router":
        if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
            return torch.mm(x.float(), state["router_t"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    if layer == "unpooler":
        n = cfg["n"]
        if n == 1024:
            w = state["unpooler_1024"]
            w_t = state["unpooler_1024_t"]
        elif n == 8192:
            w = state["unpooler_8192"]
            w_t = state["unpooler_8192_t"]
        else:
            w = state["unpooler"][:n]
            w_t = w.t().contiguous()

        if HAS_SGLANG and _bf16_gemm_dispatch_impl is not None:
            return _bf16_gemm_dispatch_impl(x, w, None)
        return torch.mm(x, w_t)

    w = state[layer]
    if HAS_SGLANG and _bf16_gemm_dispatch_impl is not None:
        return _bf16_gemm_dispatch_impl(x, w, None)
    return torch.mm(x, state[f"{layer}_t"])
"""

os.makedirs("muse_bf16_proj", exist_ok=True)
with open("muse_bf16_proj/submission.py", "w") as f:
    f.write(optimized_code)
print("✓ muse_bf16_proj/submission.py ready!")

# 5. Submit to Meta's H200 cluster
print("\\n[5/5] Submitting to Meta's H200 cluster & leaderboard...\\n")
subprocess.run(["/root/.okc/bin/okc", "submit", "muse_bf16_proj/submission.py"])
'''

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Muse Spark Open Kernel Challenge ($100k)\n",
                "### Official Submission Runner for NVIDIA H200 (Hopper SM90a)\n",
                "**Participant:** Arunjyoti Sarma (IIT BHU Research Lab)  \n",
                "**Target Problem:** `muse_bf16_proj` (Dense Projections)\n",
                "\n",
                "This notebook completely automates the entire Open Kernel Challenge workflow on Google Colab."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Verify Cloud GPU\n",
                "Click the Play button below to verify your connected remote GPU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: 1-Click Complete Pipeline (Install, Login, Deploy & Submit!)\n",
                "\n",
                "Click the Play button on the cell below:\n",
                "1. It will install `okc` CLI.\n",
                "2. It will display your 8-character device code in the output.\n",
                "3. Switch to your **[github.com/login/device](https://github.com/login/device)** tab, enter the code, and click **Authorize**.\n",
                "4. The cell will automatically detect your login, deploy our optimized zero-allocation Hopper H200 kernel into `muse_bf16_proj/submission.py`, and submit directly to Meta's 8× H200 SXM cluster!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                all_in_one_code
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(proj_root, "colab_starter_notebook.ipynb")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Generated all-in-one clean notebook at:", notebook_path)
