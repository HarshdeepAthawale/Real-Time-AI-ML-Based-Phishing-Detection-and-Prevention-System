FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p models/gnn-domain-classifier-v1

# Expose port
EXPOSE 8000

# Run application
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
