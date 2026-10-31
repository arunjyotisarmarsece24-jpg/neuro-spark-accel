#!/usr/bin/env bash
# ==============================================================================
# Helper Script for Open Kernel Challenge (OKC)
# Target: Muse Spark Open Model Optimization (H200 SXM)
# ==============================================================================

set -e

OKC_BIN="$HOME/.okc/bin/okc"

install_okc() {
    echo "==> Installing okc CLI..."
    curl -fsSL https://openkernelchallenge.dev/install.sh | sh
    export PATH="$HOME/.okc/bin:$PATH"
    echo "==> Installation complete!"
}

check_cli() {
    if [ ! -f "$OKC_BIN" ] && ! command -v okc >/dev/null 2>&1; then
        echo "okc CLI not found. Running installer..."
        install_okc
    fi
}

case "$1" in
    install)
        install_okc
        ;;
    login)
        check_cli
        okc login
        ;;
    problems)
        check_cli
        okc problems
        ;;
    init)
        check_cli
        if [ -z "$2" ]; then
            echo "Usage: ./scripts/run_okc.sh init <problem_name>"
            echo "Examples:"
            echo "  ./scripts/run_okc.sh init muse_dense_projections"
            echo "  ./scripts/run_okc.sh init muse_attention_decode"
            echo "  ./scripts/run_okc.sh init muse_routed_experts"
            echo "  ./scripts/run_okc.sh init muse_decode_layer"
            echo "  ./scripts/run_okc.sh init muse_prefill_layer"
            exit 1
        fi
        okc init "$2"
        ;;
    check)
        check_cli
        okc check
        ;;
    submit)
        check_cli
        if [ -z "$2" ]; then
            okc submit submission.py
        else
            okc submit "$2"
        fi
        ;;
    *)
        echo "Open Kernel Challenge Automation"
        echo "Usage: ./scripts/run_okc.sh {install|login|problems|init <problem>|check|submit <file>}"
        exit 1
        ;;
esac
