# Use an official lightweight Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system dependencies required for OpenCV, FFmpeg, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt imageio-ffmpeg

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:$PORT"]
