FROM python:3.11-slim

WORKDIR /app

# Chromium for Kaleido/Plotly PNG export
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium && \
    rm -rf /var/lib/apt/lists/*
ENV CHROME_PATH=/usr/bin/chromium

# harden pip against slow/flaky downloads
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY pyproject.toml .

# --- GPU-enabled ML stack (RTX 4090, CUDA 12.x) ----------------------------
# torch 2.6.0 (cu124) pins nvidia-cudnn-cu12==9.1.0.70 exactly, while
# tensorflow[and-cuda]==2.20.0 needs cuDNN >= 9.3 -> the two GPU builds cannot
# be co-resolved by pip in one environment. Resolution:
#   1. Install core + service deps (no torch/TF pulled here).
#   2. tensorflow[and-cuda] brings the (newer) NVIDIA CUDA pip wheels
#      (cuDNN 9.x, cuBLAS 12.x) that TF 2.20 requires.
#   3. torch with --no-deps so it does NOT drag in its own exact nvidia-* pins;
#      torch then loads the CUDA libs already present from TF (cuDNN is backward
#      compatible within major version 9). Its pure-python runtime deps are
#      installed explicitly afterwards.
# Both frameworks share ONE set of CUDA userspace libs; only the driver comes
# from the host via the compose GPU reservation.
RUN pip install --no-cache-dir ".[services]"

RUN pip install --no-cache-dir \
        "tensorflow[and-cuda]==2.20.0" \
        "keras==3.14.0" \
        "optuna==4.8.0"

RUN pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.6.0"

# torch's CUDA libs that tensorflow[and-cuda] does NOT provide. Everything else
# torch needs (cublas, cudnn, cufft, curand, cusolver, cusparse, nccl, nvrtc,
# nvjitlink, cupti, cuda-runtime) is supplied by TF at a newer, ABI-compatible
# version (CUDA 12.9 / cuDNN 9.24 vs torch's build-time 12.4 / 9.1).
RUN pip install --no-cache-dir \
        "nvidia-cusparselt-cu12==0.6.2" \
        "nvidia-nvtx-cu12==12.4.127" \
        "triton==3.2.0"

RUN pip install --no-cache-dir \
        "filelock" "typing-extensions>=4.10.0" "networkx" "jinja2" \
        "fsspec" "sympy==1.13.1"

COPY config/ config/
COPY src/ src/
COPY services/__init__.py services/__init__.py
COPY services/logging_config.py services/logging_config.py
COPY services/warnings_config.py services/warnings_config.py
COPY services/data_service/ services/data_service/
COPY services/model_service/ services/model_service/

ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "services.model_service.main:app", "--host", "0.0.0.0", "--port", "8002"]