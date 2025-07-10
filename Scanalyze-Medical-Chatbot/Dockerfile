# Use official Python image
FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y git \
    build-essential \
    git \
    curl \
    cmake \
    libgl1 \
    ninja-build \
    libglib2.0-0 \
    libopenblas-dev \
    && useradd -m appuser \
    && rm -rf /var/lib/apt/lists/*
    && apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
