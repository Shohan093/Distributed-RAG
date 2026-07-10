#!/usr/bin/env bash
# Runs on the Deep Learning VM image, which already has Python + PyTorch +
# CUDA + the NVIDIA driver baked in (that's the whole point of using this
# image family instead of a bare Ubuntu image + manual driver install --
# see ../env.sh's GPU_IMAGE_FAMILY comment). Only adds the few packages
# driftdrag's embedding path needs that aren't already in the image.
set -euo pipefail
exec > >(tee /var/log/driftdrag-gpu-setup.log) 2>&1

/opt/conda/bin/pip install --quiet sentence-transformers einops gcsfs redis faiss-cpu

echo "GPU VM ready. Verify with:"
echo "  /opt/conda/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
