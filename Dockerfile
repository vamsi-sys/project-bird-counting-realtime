FROM python:3.13-slim

# System deps: ffmpeg for H.264 re-encoding, OpenCV headless deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — CPU-only torch + onnxruntime
# This installs ~700MB vs ~2GB for full CUDA torch
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Ensure runtime dirs exist
RUN mkdir -p uploads outputs static

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ONNXRUNTIME_DISABLE_GPU=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
