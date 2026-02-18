# Production Dockerfile for VeriFact API
# ML models (NLI + SBERT) are pre-downloaded during build
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set NLTK data path
ENV NLTK_DATA=/usr/local/share/nltk_data

# Create directory and download NLTK data
RUN mkdir -p /usr/local/share/nltk_data && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')"

# Pre-download ML models during build (avoids runtime download failures + cold-start delays)
# Models are cached in /root/.cache/huggingface and later moved to appuser's home
RUN python -c "\
    from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small'); \
    AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small'); \
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    print('All models pre-downloaded successfully')"

# Copy application code
COPY . .

# Create non-root user for security
# Move cached models to appuser's home so they're accessible at runtime
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache && \
    cp -r /root/.cache/huggingface /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Expose port
EXPOSE 5000

# Health check — models are pre-loaded, generous start-period for first inference
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run with gunicorn
# --workers 1: single worker to avoid duplicate model loading (each worker = separate process = separate copy of ~280MB models)
# --threads 4: concurrency via threading (models are shared within process)
# --timeout 180: generous timeout for first-request model initialization
# --max-requests 500: periodic worker restart for garbage collection
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "--max-requests", "500", "--max-requests-jitter", "50", "app_flask:app"]
