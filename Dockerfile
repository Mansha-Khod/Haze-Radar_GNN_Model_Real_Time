# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU-only (smallest version)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric (minimal, will use CPU wheels)
RUN pip install --no-cache-dir torch-geometric

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY artifacts/ artifacts/

# Clean up to reduce image size
RUN apt-get purge -y --auto-remove gcc g++ && \
    rm -rf /root/.cache/pip

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
