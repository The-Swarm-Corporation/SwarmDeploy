# Use the official Python 3.12 image
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies for faster builds
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout and retry
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=5 -r requirements.txt

# Copy the application code
FROM base
COPY --from=builder /app /app
COPY . .

# Command to run the application with dynamic worker count
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 --workers $(nproc)"]