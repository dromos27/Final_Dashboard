#!/usr/bin/env bash
# ============================================================================
# render-build.sh — Build script for Render deployment
# ============================================================================
set -o errexit   # exit on error

echo "Installing system dependencies"
# opencv-python-headless avoids needing GUI libs on a headless server
pip install --upgrade pip

echo "Installing Python packages"
# Install CPU-only PyTorch first (saves ~1.5 GB vs. CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (torch/torchvision already satisfied)
pip install -r requirements.txt

# Replace opencv-python with headless variant (no X11/GUI deps needed)
pip uninstall -y opencv-python 2>/dev/null || true
pip install opencv-python-headless

echo "Creating required directories"
mkdir -p exports assets

echo "Build complete"
