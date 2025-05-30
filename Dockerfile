# Use an official Python runtime
FROM python:3.9-slim

# Install system dependencies including X11 and other required libraries for Open3D
RUN apt-get update && apt-get install -y \
    netcat-traditional \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app

CMD ["./entrypoint.sh"]