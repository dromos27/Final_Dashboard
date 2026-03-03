#!/usr/bin/env bash
# ============================================================================
# render-build.sh — Build script for Render free-tier deployment
# Uses requirements-render.txt (no PyTorch/YOLO/OpenCV) to stay under 512 MB
# ============================================================================
set -o errexit   # exit on error

echo "Installing slim cloud dependencies"
pip install --upgrade pip
pip install -r requirements-render.txt

echo "Creating required directories"
mkdir -p exports assets

echo "Build complete"
