FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir torch-geometric

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY artifacts/ artifacts/

RUN apt-get purge -y --auto-remove gcc g++ && \
    rm -rf /root/.cache/pip

EXPOSE 8000

CMD ["python", "main.py"]

