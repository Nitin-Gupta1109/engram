FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY engram/ engram/

# Install engram with cloud + server deps
RUN pip install --no-cache-dir -e ".[cloud]" fastapi uvicorn

# Pre-download the default embedding model
RUN python -c "from engram.retrieval.embedder import Embedder; Embedder('bge-large')._load()"

EXPOSE 8000

ENV ENGRAM_BACKEND=faiss
ENV ENGRAM_STORE_PATH=/data/engram_store

VOLUME ["/data"]

CMD ["uvicorn", "engram.server:app", "--host", "0.0.0.0", "--port", "8000"]
