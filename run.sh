#!/bin/bash
set -e

echo "=== [1/4] Ensuring OKC CLI is installed ==="
OKC_BIN=""
if command -v okc >/dev/null 2>&1; then
    OKC_BIN="okc"
elif [ -x "$HOME/.okc/bin/okc" ]; then
    OKC_BIN="$HOME/.okc/bin/okc"
elif [ -x "/root/.okc/bin/okc" ]; then
    OKC_BIN="/root/.okc/bin/okc"
else
    echo "Installing okc CLI..."
    curl -fsSL https://openkernelchallenge.dev/install.sh | sh
    if [ -x "$HOME/.okc/bin/okc" ]; then
        OKC_BIN="$HOME/.okc/bin/okc"
    elif [ -x "/root/.okc/bin/okc" ]; then
        OKC_BIN="/root/.okc/bin/okc"
    else
        OKC_BIN="okc"
    fi
fi

export PATH="$HOME/.okc/bin:/root/.okc/bin:$PATH"
echo "✓ OKC CLI found: $OKC_BIN"

echo "=== [2/4] Preparing workspace ==="
mkdir -p muse_bf16_proj

echo "=== [3/4] Downloading latest 199 KB Beast Kernel (commit c6faf91) ==="
curl -fsSL https://raw.githubusercontent.com/arunjyotisarmarsece24-jpg/neuro-spark-accel/main/colab_submission.py -o muse_bf16_proj/submission.py

SIZE=$(wc -c < muse_bf16_proj/submission.py)
echo "✓ Kernel ready! Size: $SIZE bytes"

echo "=== [4/4] Submitting to Meta H200 Cluster ==="
"$OKC_BIN" submit muse_bf16_proj/submission.py
