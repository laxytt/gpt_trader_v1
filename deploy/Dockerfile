# Multi-stage Dockerfile for GPT Trading System
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trader

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/trader/.local

# Copy application code
COPY --chown=trader:trader . .

# Create necessary directories
RUN mkdir -p logs data screenshots models && \
    chown -R trader:trader logs data screenshots models

# Switch to non-root user
USER trader

# Add local bin to PATH
ENV PATH=/home/trader/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "trading_loop.py"]