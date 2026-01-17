FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU version first (largest dependency)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY main.py .
COPY artifacts/ artifacts/

# Clean up build dependencies to reduce image size
RUN apt-get purge -y --auto-remove gcc g++ && \
    rm -rf /root/.cache/pip

# Expose port (Railway will override this with $PORT)
EXPOSE 8000

# Use environment variable directly in Python
CMD ["python", "main.py"]
