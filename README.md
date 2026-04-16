<p align="center">
  <img src="logo.svg" width="180" alt="Engram logo">
</p>

<h1 align="center">Engram</h1>

<p align="center">
  <strong>High-recall conversational memory retrieval. Local-first, cloud-ready.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/engram-search/"><img src="https://img.shields.io/pypi/v/engram-search?color=blue" alt="PyPI"></a>
  <a href="https://github.com/Nitin-Gupta1109/engram/actions"><img src="https://github.com/Nitin-Gupta1109/engram/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Nitin-Gupta1109/engram/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://pypi.org/project/engram-search/"><img src="https://img.shields.io/pypi/pyversions/engram-search" alt="Python"></a>
</p>

---

## Benchmark Results

**98.4% R@5 on LongMemEval** (500 questions) — no LLM required, zero cost per query.

| Metric | Score |
|--------|-------|
| R@5 | **98.4%** (492/500) |
| R@10 | 99.4% |
| NDCG@5 | 0.934 |

| Question Type | R@5 |
|--------------|-----|
| knowledge-update | 98.7% |
| multi-session | 99.2% |
| single-session-assistant | 100.0% |
| single-session-user | 100.0% |
| temporal-reasoning | 97.0% |
| single-session-preference | 93.3% |

## What It Does

Engram stores conversation history and retrieves it with state-of-the-art accuracy. It uses a three-stage retrieval pipeline — dense embeddings, sparse keyword matching, and cross-encoder reranking — to achieve higher recall than systems relying on LLM-based extraction or summarization.

Nothing is summarized. Nothing is paraphrased. Your exact words are stored and returned.

## How It Compares

| | Engram | MemPalace | Mem0 |
|---|---|---|---|
| **R@5 (LongMemEval)** | **98.4%** | 96.6% | — |
| Embedding model | bge-large (1024d) | all-MiniLM (384d) | Varies |
| Sparse retrieval | BM25 + RRF fusion | Ad-hoc keyword overlap | N/A |
| Reranking | Cross-encoder (free) | LLM call ($0.001/q) | N/A |
| Indexing | User + assistant + preference docs | User turns only | LLM-extracted facts |
| Cloud deployment | Qdrant backend | No | Yes |
| LLM required | **No** | No (optional rerank) | Yes |

## Install

```bash
pip install engram-search
```

Optional extras:

```bash
# With cloud backend (Qdrant)
pip install engram-search[cloud]

# With cross-encoder reranker
pip install engram-search[rerank]

# Everything (dev + cloud + rerank)
pip install engram-search[all]
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

## Quickstart — Cloud Mode

```bash
# Set up Qdrant (managed or self-hosted)
export ENGRAM_BACKEND=qdrant
export ENGRAM_QDRANT_URL=https://your-cluster.qdrant.io:6333
export ENGRAM_QDRANT_API_KEY=your-api-key

# Start the API server
pip install fastapi uvicorn
uvicorn engram.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Add conversations |
| `POST` | `/search` | Search memories |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Store statistics |

## Examples

Check out the interactive notebooks in [`examples/`](examples/):

| Notebook | Description |
|----------|-------------|
| [Getting Started](examples/01_getting_started.ipynb) | Ingest conversations, search memories, understand hybrid retrieval |
| [Customer Support](examples/02_customer_support.ipynb) | Build a support agent with full customer history recall |
| [Personal Assistant](examples/03_personal_assistant.ipynb) | AI assistant with long-term memory across conversations |

## Docker

```bash
# Local mode
docker compose up

# Or build and run directly
docker build -t engram .
docker run -p 8000:8000 -v engram_data:/data engram
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

## Run Benchmarks

```bash
# Download LongMemEval dataset
curl -fsSL -o /tmp/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Install dev dependencies
pip install engram-search[all]

# Hybrid mode (default — dense + BM25 via RRF)
python benchmarks/longmemeval_bench.py /tmp/longmemeval_s_cleaned.json --mode hybrid

# Dense only
python benchmarks/longmemeval_bench.py /tmp/longmemeval_s_cleaned.json --mode dense

# Full pipeline (hybrid + cross-encoder reranker)
python benchmarks/longmemeval_bench.py /tmp/longmemeval_s_cleaned.json --mode rerank
```

## Requirements

- Python 3.9+
- ~1.3 GB disk for bge-large embedding model (downloaded on first use)
- No API keys required for local mode

## License

MIT
