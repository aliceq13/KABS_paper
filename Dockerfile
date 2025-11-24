# Dockerfile for KABS Enhance - Dual Re-ID System
# Based on PyTorch with CUDA support

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
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
    h5py

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data /workspace/output

# Copy application code
COPY . /workspace/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
