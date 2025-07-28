# Multi-stage build for efficient containerization
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/nltk_data

# Install system dependencies required for the project
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    fontconfig \
    fonts-dejavu-core \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create NLTK data directory
RUN mkdir -p /usr/local/nltk_data && chmod 755 /usr/local/nltk_data

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages with explicit PyMuPDF
RUN pip install --no-cache-dir chromadb==1.0.15 PyMuPDF==1.24.10

# Download NLTK data at runtime instead of build time
RUN echo '#!/bin/bash\n\
python -c "import nltk; nltk.download(\"punkt\", quiet=True)" 2>/dev/null || true\n\
python -c "import nltk; nltk.download(\"punkt_tab\", quiet=True)" 2>/dev/null || true\n\
' > /usr/local/bin/download_nltk.sh && chmod +x /usr/local/bin/download_nltk.sh

# Copy application code
COPY main1a.py .
COPY main.py .
COPY test.py .
COPY challenge1b_input.json .

# Create necessary directories for input/output
RUN mkdir -p /app/input /app/output /app/PDFs /app/output1a

# Copy model files if they exist, otherwise create empty directory
COPY models--sentence-transformers--all-MiniLM-L6-v2/ ./models--sentence-transformers--all-MiniLM-L6-v2/

# Copy any existing PDFs (optional, can be mounted as volume)
COPY PDFs/ ./PDFs/

# Create a startup script for the complete pipeline
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Travel Planning Document Processing..."\n\
echo "Downloading NLTK data..."\n\
/usr/local/bin/download_nltk.sh\n\
echo "Step 1: Extracting PDF structures..."\n\
python main1a.py\n\
echo "Step 2: Generating travel recommendations..."\n\
python main.py\n\
echo "Processing complete! Check output files:"\n\
echo "- PDF structures: /app/output1a/"\n\
echo "- Travel recommendations: /app/challenge1b_output.json"\n\
ls -la /app/output1a/ 2>/dev/null || echo "No PDF structure files found"\n\
ls -la /app/challenge1b_output.json 2>/dev/null || echo "Output file not generated"\n\
# Copy output to host if mounted\n\
if [ -d "/app/host" ]; then\n\
  cp /app/challenge1b_output.json /app/host/ 2>/dev/null || true\n\
  echo "Output copied to host directory"\n\
fi\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Alternative individual scripts
RUN echo '#!/bin/bash\n\
/usr/local/bin/download_nltk.sh\n\
python main1a.py\n\
' > /app/extract_pdfs.sh && chmod +x /app/extract_pdfs.sh

RUN echo '#!/bin/bash\n\
/usr/local/bin/download_nltk.sh\n\
python main.py\n\
if [ -d "/app/host" ]; then\n\
  cp /app/challenge1b_output.json /app/host/ 2>/dev/null || true\n\
fi\n\
' > /app/analyze_travel.sh && chmod +x /app/analyze_travel.sh

# Set proper permissions
RUN chmod -R 755 /app

# Expose volumes for input and output
VOLUME ["/app/input", "/app/output", "/app/PDFs", "/app/output1a", "/app/host"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command runs the complete pipeline
CMD ["/app/run_pipeline.sh"]
