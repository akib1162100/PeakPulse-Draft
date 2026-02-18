FROM python:3.11-slim AS base

# System dependencies for matplotlib, reportlab, and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .

# Install CPU-only PyTorch from official index, then remaining requirements
RUN pip install --no-cache-dir --timeout 120 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 120 -r requirements.txt

# Pre-download the MiniLM transformer model at build time (~80MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model cached')"

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/

# Create data directory for uploads/outputs
RUN mkdir -p /app/data /app/tmp

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
