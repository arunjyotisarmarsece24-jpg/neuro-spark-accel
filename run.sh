#!/bin/bash
set -e

echo "=== [1/3] Preparing workspace ==="
mkdir -p muse_bf16_proj

echo "=== [2/3] Downloading latest 199 KB Beast Kernel (commit c6faf91) ==="
curl -fsSL https://raw.githubusercontent.com/arunjyotisarmarsece24-jpg/neuro-spark-accel/main/colab_submission.py -o muse_bf16_proj/submission.py

SIZE=$(wc -c < muse_bf16_proj/submission.py)
echo "✓ Kernel ready! Size: $SIZE bytes"

echo "=== [3/3] Submitting to Meta H200 Cluster ==="
/root/.okc/bin/okc submit muse_bf16_proj/submission.py
