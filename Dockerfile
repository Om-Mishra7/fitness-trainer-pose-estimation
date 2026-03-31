FROM python:3.11-slim

# System dependencies for OpenCV, MediaPipe, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the pose landmarker model at build time so the container starts instantly
RUN wget -q -O pose_landmarker_full.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# Copy application code
COPY . .

# Uploads directory
RUN mkdir -p uploads

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Note: webcam-based live tracking requires --device /dev/video0 at runtime.
# Video file analysis works without any special flags.
EXPOSE 5000

CMD ["python", "app.py"]
