# Production Dockerfile for VeriFact API
# Lightweight — no local ML models (uses HF Inference API)
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

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check — fast startup, no model loading delay
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run with gunicorn
# --workers 2: safe now (no 1.6GB model per worker)
# --threads 4: concurrency
# --timeout 120: API calls are fast, no local model loading
# --max-requests 500: periodic worker restart for garbage collection
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "--max-requests", "500", "--max-requests-jitter", "50", "app_flask:app"]
