FROM python:3.9-slim

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies with increased timeout and retries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 1000 --retries 10 -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories for uploads and models
RUN mkdir -p uploads models

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
