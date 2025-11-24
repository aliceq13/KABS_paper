# Dockerfile for KABS Enhance - Dual Re-ID System
# Based on PyTorch with CUDA support

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during build (e.g. timezone selection)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Added build-essential (gcc/g++) for compiling FastReID extensions
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    g++ \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Added cython, faiss-gpu, and tabulate for FastReID
RUN pip install --no-cache-dir \
    opencv-python \
    ultralytics \
    scipy \
    matplotlib \
    pillow \
    torchreid \
    transformers \
    yacs \
    termcolor \
    tb-nightly \
    future \
    h5py \
    cython \
    faiss-gpu \
    tabulate \
    gdown

# Fix NumPy and Transformers version compatibility with PyTorch 2.1
# Must be done AFTER initial install to override base image versions
RUN pip uninstall -y numpy transformers && \
    pip install --no-cache-dir "numpy<2.0" "transformers<4.38.0"

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data /workspace/output

# Copy application code
COPY . /workspace/

# Install FastReID from local clone
# Removed pip install -e . as setup.py is missing
# Instead, we add it to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/workspace/fast-reid"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
