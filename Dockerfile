FROM python:3.13-slim

# Install system dependencies including ffmpeg (required for H.264 re-encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -1

WORKDIR /app

# Install Python deps first (layer cache)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all backend source
COPY backend/ ./

# Ensure runtime dirs exist
RUN mkdir -p uploads outputs static

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
