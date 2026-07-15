#!/usr/bin/env bash
# Bootstrap spriteforge on a fresh rented CUDA box (RunPod / Vast / Lambda).
#
#   export SPRITEFORGE_REPO=https://<token>@github.com/J-Glatter/surfside-code-sample
#   curl -fsSL <raw url of this file> | bash
#   # or: bash scripts/remote_gpu_setup.sh   (after a manual clone)
#
# Installs into the pod's system python on purpose — GPU-cloud PyTorch
# templates ship a CUDA-matched torch there, and a plain venv would hide it.
set -euo pipefail

REPO="${SPRITEFORGE_REPO:-https://github.com/J-Glatter/surfside-code-sample}"
BASE="${SPRITEFORGE_BASE:-/workspace}"
[ -d "$BASE" ] || BASE="$HOME"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "no GPU visible — wrong pod template?"; exit 1; }

echo "== clone =="
cd "$BASE"
if [ -d spriteforge/.git ]; then
    git -C spriteforge pull --ff-only
else
    git clone "$REPO" spriteforge
fi
cd spriteforge

echo "== torch =="
python3 - <<'EOF' || pip install torch --index-url https://download.pytorch.org/whl/cu124
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
EOF

echo "== spriteforge =="
pip install -q -e ".[generate,curate,animate,director]"

python3 -c "import torch; print('cuda:', torch.cuda.is_available(),
'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
spriteforge --help | head -3

if [ "${RUN_SMOKE:-0}" = "1" ]; then
    echo "== smoke test (downloads ~4 GB of SD weights on first run) =="
    spriteforge make "a small slime monster" -o "$BASE/smoke" --offline
    ls -la "$BASE/smoke"
fi

echo "== ready =="
echo "next: docs/CHECKPOINTS.md — Checkpoint A/B commands"
