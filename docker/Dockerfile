# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build stage
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS builder

# Install build dependencies and create venv in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the entire application
COPY pyproject.toml pyproject.toml
COPY swagger swagger
COPY scripts scripts
COPY maps maps
COPY LICENSE LICENSE

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    find /opt/venv -name __pycache__ -type d -exec rm -rf {} +

# Create a stage for extracting only the essential CUDA headers
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS cuda-minimal
RUN mkdir -p /cuda-minimal/include && \
    cp -r /usr/local/cuda/include/* /cuda-minimal/include/ && \
    # Keep only essential CUDA dev libraries
    mkdir -p /cuda-minimal/lib64 && \
    cp /usr/local/cuda/lib64/libcudart.so* /cuda-minimal/lib64/ && \
    cp /usr/local/cuda/lib64/libnvrtc* /cuda-minimal/lib64/

# Final stage - using runtime image
FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

# Install regular Python and other runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-distutils \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only the necessary CUDA headers and libraries
COPY --from=cuda-minimal /cuda-minimal/include /usr/local/cuda/include
COPY --from=cuda-minimal /cuda-minimal/lib64 /usr/local/cuda/lib64

# Copy the virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy the whole application
COPY --from=builder /app /app

# Create directories with appropriate permissions
RUN mkdir -p /app/graphs /app/results /app/visualizations && \
    touch /app/cufile.log && \
    chown -R appuser:appuser /app && \
    # Create a place for CUDA logs
    mkdir -p /tmp/nvidia && \
    chmod 777 /tmp/nvidia

# Switch to non-root user
USER appuser

# Expose port for service
EXPOSE 8000

# Default command
CMD ["python3", "scripts/rest_api.py"]
