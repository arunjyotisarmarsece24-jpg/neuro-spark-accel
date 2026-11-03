import json
import os

optimized_kernel_lines = [
    "# okc: problem=muse_bf16_proj\n",
    '"""Optimized Dense Projections for Muse Spark Open on Hopper H200.\n',
    "Author: Arunjyoti Sarma (IIT BHU Research Lab)\n",
    "\n",
    "Key Techniques:\n",
    "1. Zero-Allocation Hot Path: Pre-slice and pre-contiguous all weight buffers during untimed prepare().\n",
    "2. Pre-Transposed Layouts: Eliminate runtime matrix transpositions.\n",
    "3. Dual-Path Router: Pre-transposed FP32 buffer for tokens <= 8, split BF16 GEMM for tokens > 8.\n",
    "4. Native cuBLASLt Dispatch with Zero Slicing Overhead.\n",
    '"""\n',
    "\n",
    "from typing import TypedDict\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "\n",
    "try:\n",
    "    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl\n",
    "    HAS_SGLANG = True\n",
    "except Exception:\n",
    "    HAS_SGLANG = False\n",
    "    _bf16_gemm_dispatch_impl = None\n",
    "\n",
    "\n",
    "class Weights(TypedDict):\n",
    "    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]\n",
    "    o: torch.Tensor         # bfloat16 [8192, 1024]\n",
    "    pooler: torch.Tensor    # bfloat16 [4096, 8192]\n",
    "    unpooler: torch.Tensor  # bfloat16 [8192, 4096]\n",
    "    router: torch.Tensor    # float32 [256, 8192]\n",
    "\n",
    "\n",
    "class Config(TypedDict):\n",
    "    m: int\n",
    "    n: int\n",
    "    k: int\n",
    "    layer: str\n",
    "\n",
    "\n",
    "class Inputs(TypedDict):\n",
    "    x: torch.Tensor\n",
    "    config: Config\n",
    "\n",
    "\n",
    "ROUTER_FP32_MAX_TOKENS = 8\n",
    "\n",
    "\n",
    "def prepare(weights: Weights):\n",
    "    state = {}\n",
    "    for name, w in weights.items():\n",
    "        w_contig = w.contiguous()\n",
    "        state[name] = w_contig\n",
    '        state[f"{name}_t"] = w_contig.t().contiguous()\n',
    "\n",
    '    unpooler = weights["unpooler"]\n',
    '    state["unpooler_1024"] = unpooler[:1024].contiguous()\n',
    '    state["unpooler_1024_t"] = unpooler[:1024].t().contiguous()\n',
    '    state["unpooler_8192"] = unpooler.contiguous()\n',
    '    state["unpooler_8192_t"] = unpooler.t().contiguous()\n',
    "\n",
    '    router = weights["router"]\n',
    '    state["router_t"] = router.t().contiguous()\n',
    "    hi = router.to(torch.bfloat16)\n",
    "    lo = (router - hi.float()).to(torch.bfloat16)\n",
    '    state["router_hi_t"] = hi.t().contiguous()\n',
    '    state["router_lo_t"] = lo.t().contiguous()\n',
    "\n",
    "    return state\n",
    "\n",
    "\n",
    "def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:\n",
    '    x = data["x"]\n',
    '    cfg = data["config"]\n',
    '    layer = cfg["layer"]\n',
    "\n",
    '    if layer == "router":\n',
    "        if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:\n",
    '            return torch.mm(x.float(), state["router_t"])\n',
    '        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)\n',
    '        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)\n',
    "        return acc\n",
    "\n",
    '    if layer == "unpooler":\n',
    '        n = cfg["n"]\n',
    "        if n == 1024:\n",
    '            w = state["unpooler_1024"]\n',
    '            w_t = state["unpooler_1024_t"]\n',
    "        elif n == 8192:\n",
    '            w = state["unpooler_8192"]\n',
    '            w_t = state["unpooler_8192_t"]\n',
    "        else:\n",
    '            w = state["unpooler"][:n]\n',
    "            w_t = w.t().contiguous()\n",
    "\n",
    "        if HAS_SGLANG and _bf16_gemm_dispatch_impl is not None:\n",
    "            return _bf16_gemm_dispatch_impl(x, w, None)\n",
    "        return torch.mm(x, w_t)\n",
    "\n",
    "    w = state[layer]\n",
    "    if HAS_SGLANG and _bf16_gemm_dispatch_impl is not None:\n",
    "        return _bf16_gemm_dispatch_impl(x, w, None)\n",
    '    return torch.mm(x, state[f"{layer}_t"])\n'
]

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Muse Spark Open Kernel Challenge ($100k)\n",
                "### Official Submission Runner for NVIDIA H200 (Hopper SM90a)\n",
                "**Participant:** Arunjyoti Sarma (IIT BHU Research Lab)\n",
                "**Target Problem:** `muse_bf16_proj` (Dense Projections)\n",
                "\n",
                "Simply click the Play button **`[ ▶ ]`** on each cell from top to bottom!"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Verify Cloud GPU"
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
                "## Step 2: Install `okc` CLI & Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!curl -fsSL https://openkernelchallenge.dev/install.sh | sh\n",
                "import os\n",
                "os.environ[\"PATH\"] = \"/root/.okc/bin:\" + os.environ[\"PATH\"]\n",
                "!pip install -q triton torch"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Login to OKC with GitHub\n",
                "This uses a Python pseudo-terminal (PTY) to accept the rules and display your 8-character device code."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, pty, subprocess, time, select\n",
                "\n",
                "master, slave = pty.openpty()\n",
                "proc = subprocess.Popen(\n",
                "    ['/root/.okc/bin/okc', 'login'],\n",
                "    stdin=slave, stdout=slave, stderr=slave, close_fds=True\n",
                ")\n",
                "os.close(slave)\n",
                "\n",
                "# Wait 1 second and automatically accept terms\n",
                "time.sleep(1.0)\n",
                "os.write(master, b'y\\n')\n",
                "\n",
                "print('=== OKC GITHUB LOGIN ===')\n",
                "while True:\n",
                "    r, _, _ = select.select([master], [], [], 2.0)\n",
                "    if r:\n",
                "        try:\n",
                "            data = os.read(master, 1024)\n",
                "            if not data:\n",
                "                break\n",
                "            print(data.decode('utf-8', errors='ignore'), end='', flush=True)\n",
                "        except OSError:\n",
                "            break\n",
                "    elif proc.poll() is not None:\n",
                "        break\n",
                "\n",
                "proc.wait()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Initialize Problem Workspace (`muse_bf16_proj`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!okc init muse_bf16_proj"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Deploy Our Optimized Hopper H200 Kernel to `muse_bf16_proj/submission.py`\n",
                "This pre-packs weights and eliminates runtime slicing to beat SGLang's cuBLAS baseline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "code = '''" + "".join(optimized_kernel_lines) + "'''\n",
                "import os\n",
                "os.makedirs('muse_bf16_proj', exist_ok=True)\n",
                "with open('muse_bf16_proj/submission.py', 'w') as f:\n",
                "    f.write(code)\n",
                "print('✓ Successfully deployed optimized kernel to muse_bf16_proj/submission.py!')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Submit to Meta's Official H200 Cluster & Leaderboard! 🚀"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!okc submit muse_bf16_proj/submission.py"
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

print("Generated clean 1-click notebook at:", notebook_path)
