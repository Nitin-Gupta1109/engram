# Engram

High-recall conversational memory retrieval. Local-first, cloud-ready.

**Target: 98%+ R@5 on LongMemEval — no LLM required.**

## What It Is

Engram stores conversation history and retrieves it with state-of-the-art accuracy. It uses a three-stage retrieval pipeline — dense embeddings, sparse keyword matching, and cross-encoder reranking — to achieve higher recall than systems relying on LLM-based extraction or summarization.

Nothing is summarized. Nothing is paraphrased. Your exact words are stored and returned.

## Why It's Better

| | Engram | MemPalace | Mem0 |
|---|---|---|---|
| Embedding model | bge-large (1024d) | all-MiniLM (384d) | Varies |
| Sparse retrieval | BM25 with RRF fusion | Ad-hoc keyword overlap | N/A |
| Reranking | Cross-encoder (free) | LLM call ($0.001/q) | N/A |
| Indexing | User + assistant turns | User turns only | LLM-extracted facts |
| Cloud deployment | Qdrant backend | No | Yes |
| LLM required | No | No (optional rerank) | Yes |

## Install

```bash
# Local mode (FAISS + SQLite)
pip install engram

# With cloud backend (Qdrant)
pip install engram[cloud]

# With cross-encoder reranker
pip install engram[rerank]

# Everything
pip install engram[all]
```

## Quickstart — Local Mode

```bash
# Initialize a memory store
engram init ./my_memories

# Ingest conversations
engram ingest conversations.json --store ./my_memories

# Search
engram search "why did we switch to GraphQL" --store ./my_memories
```

## Quickstart — Cloud Mode (for companies)

```bash
# Set up Qdrant (managed or self-hosted)
export ENGRAM_BACKEND=qdrant
export ENGRAM_QDRANT_URL=https://your-cluster.qdrant.io:6333
export ENGRAM_QDRANT_API_KEY=your-api-key

# Start the API server
pip install fastapi uvicorn
uvicorn engram.server:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /ingest` — add conversations
- `POST /search` — search memories
- `GET /health` — health check
- `GET /stats` — store statistics

## Benchmarks

```bash
# Download LongMemEval dataset
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Run benchmark
pip install -e ".[dev]"

# Dense only (bge-large baseline)
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode dense

# Hybrid (dense + BM25 via RRF) — default
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode hybrid

# Full pipeline (hybrid + cross-encoder reranker)
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode rerank
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Engram                               │
│                                                             │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────────┐    │
│  │ Ingestion  │  │   Index     │  │    Retrieval      │    │
│  │            │→ │             │→ │                   │    │
│  │ user+asst  │  │ FAISS (local│  │ 1. Dense (bi-enc) │    │
│  │ turns      │  │  or Qdrant  │  │ 2. BM25 (sparse)  │    │
│  │ preference │  │ (cloud)     │  │ 3. RRF fusion     │    │
│  │ extraction │  │             │  │ 4. Cross-encoder   │    │
│  └────────────┘  └─────────────┘  └───────────────────┘    │
│                                                             │
│  Local: FAISS + SQLite    Cloud: Qdrant + REST API          │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- ~1.3 GB disk for bge-large embedding model (downloaded on first use)
- No API key required for local mode

## License

MIT
