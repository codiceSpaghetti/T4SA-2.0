#!/bin/bash

# Update package list
apt-get update

# Install system dependencies for OpenCV and video processing
apt-get install -y ffmpeg libsm6 libxext6

# Install CUDA dependencies (optional, uncomment if needed)
# apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2

echo "System dependencies installed successfully!"