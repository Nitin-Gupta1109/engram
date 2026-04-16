#!/usr/bin/env python3
"""
Engram x LongMemEval Benchmark
================================

Evaluates Engram's retrieval against the LongMemEval benchmark.

For each of the 500 questions:
1. Ingest all haystack sessions as documents (user+assistant turns)
2. Encode with bge-large-en-v1.5 (1024-dim)
3. Retrieve via dense + BM25 + RRF fusion
4. Optionally rerank with cross-encoder (no LLM needed)
5. Score retrieval against ground-truth answer sessions

Modes:
    dense       — bge-large dense retrieval only
    hybrid      — dense + BM25 via RRF fusion (default)
    rerank      — hybrid + cross-encoder reranking
    full        — hybrid + rerank + preference docs + assistant turns

Usage:
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode rerank
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode full
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --limit 20
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# =============================================================================
# METRICS
# =============================================================================


def dcg(relevances, k):
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved_ids, correct_ids, k):
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def recall_at_k(retrieved_ids, correct_ids, k):
    top_k = set(retrieved_ids[:k])
    return float(any(cid in top_k for cid in correct_ids))


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_benchmark(
    data_path: str,
    mode: str = "hybrid",
    embed_model: str = "bge-large",
    limit: int = 0,
    top_k: int = 5,
    include_assistant: bool = True,
    use_prefs: bool = True,
    use_reranker: bool = False,
):
    """Run the LongMemEval benchmark."""
    from engram.retrieval.embedder import Embedder
    from engram.retrieval.sparse import BM25
    from engram.retrieval.pipeline import (
        RetrievalPipeline,
        reciprocal_rank_fusion,
        extract_person_names,
        extract_quoted_phrases,
        parse_temporal_offset,
        parse_date,
    )
    from engram.ingestion.parser import session_to_documents
    from engram.backends.base import Document

    # Load dataset — stream to avoid holding full 265MB+ JSON in memory
    print(f"Loading dataset from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    total_questions = len(data)
    if limit > 0:
        data = data[:limit]
    print(f"Loaded {total_questions} questions, using {len(data)}")

    # Free memory: we only need the subset
    import gc
    gc.collect()

    # Initialize embedder
    print(f"Initializing embedder: {embed_model}...")
    embedder = Embedder(embed_model)

    # Initialize reranker if needed
    reranker = None
    if use_reranker or mode in ("rerank", "full"):
        print("Initializing cross-encoder reranker...")
        from engram.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    # Initialize pipeline
    pipeline = RetrievalPipeline(
        embedder=embedder,
        reranker=reranker,
        use_reranker=(use_reranker or mode in ("rerank", "full")),
        dense_top_k=50,
    )

    # Results tracking
    results = []
    type_results = defaultdict(list)
    total_time = 0.0

    print(f"\nRunning benchmark: mode={mode}, embed={embed_model}, top_k={top_k}")
    print(f"Include assistant turns: {include_assistant}")
    print(f"Preference docs: {use_prefs}")
    print(f"Cross-encoder reranker: {use_reranker or mode in ('rerank', 'full')}")
    print("=" * 60)

    for qi, entry in enumerate(data):
        question = entry["question"]
        q_type = entry.get("question_type", "unknown")
        q_date = entry.get("question_date", "")
        correct_session_ids = set(entry.get("answer_session_ids", []))

        sessions = entry["haystack_sessions"]
        session_ids = entry["haystack_session_ids"]
        dates = entry["haystack_dates"]

        t0 = time.time()

        # --- Ingest ---
        # Convert sessions to documents
        all_docs_raw = []
        session_id_for_doc = {}  # doc_id -> session_id (for evaluation)

        for session, sess_id, date in zip(sessions, session_ids, dates):
            parsed = session_to_documents(
                session=session,
                session_id=sess_id,
                timestamp=date,
                include_assistant=include_assistant,
                generate_preference_doc=use_prefs,
            )
            for doc_info in parsed:
                all_docs_raw.append(doc_info)
                session_id_for_doc[doc_info["id"]] = sess_id

        if not all_docs_raw:
            results.append({"recall@5": 0, "recall@10": 0})
            continue

        # Embed all documents
        texts = [d["text"] for d in all_docs_raw]
        embeddings = embedder.encode_documents(texts)

        documents = []
        for i, doc_info in enumerate(all_docs_raw):
            documents.append(Document(
                id=doc_info["id"],
                text=doc_info["text"],
                embedding=embeddings[i].tolist(),
                metadata=doc_info["metadata"],
            ))

        # --- Retrieve ---
        if mode == "dense":
            # Dense only — no BM25, no reranker
            query_vec = embedder.encode_query(question)
            scored = []
            for doc in documents:
                emb = np.array(doc.embedding)
                sim = float(np.dot(query_vec, emb))
                scored.append((doc, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved = [d for d, _ in scored[:top_k * 2]]
        else:
            # Full pipeline (hybrid, rerank, or full mode)
            retrieved = pipeline.search(
                query=question,
                documents=documents,
                top_k=top_k * 2,  # over-fetch for evaluation at multiple k
                question_date=q_date,
            )

        elapsed = time.time() - t0
        total_time += elapsed

        # --- Evaluate ---
        # Map retrieved doc IDs back to session IDs
        retrieved_session_ids = []
        seen_sessions = set()
        for doc in retrieved:
            sess_id = session_id_for_doc.get(doc.id, doc.id)
            if sess_id not in seen_sessions:
                seen_sessions.add(sess_id)
                retrieved_session_ids.append(sess_id)

        r5 = recall_at_k(retrieved_session_ids, correct_session_ids, 5)
        r10 = recall_at_k(retrieved_session_ids, correct_session_ids, 10)
        n5 = ndcg_at_k(retrieved_session_ids, correct_session_ids, 5)
        n10 = ndcg_at_k(retrieved_session_ids, correct_session_ids, 10)

        result = {
            "question_id": entry.get("question_id", f"q_{qi}"),
            "question_type": q_type,
            "recall@5": r5,
            "recall@10": r10,
            "ndcg@5": n5,
            "ndcg@10": n10,
            "time": round(elapsed, 3),
            "hit@5": r5 > 0,
        }
        results.append(result)
        type_results[q_type].append(result)

        # Progress
        if (qi + 1) % 10 == 0 or qi == len(data) - 1:
            running_r5 = sum(r["recall@5"] for r in results) / len(results)
            print(
                f"  [{qi+1}/{len(data)}] R@5={running_r5:.1%} "
                f"(this: {'HIT' if r5 > 0 else 'MISS'}) "
                f"[{elapsed:.1f}s]"
            )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    avg_r5 = sum(r["recall@5"] for r in results) / len(results)
    avg_r10 = sum(r["recall@10"] for r in results) / len(results)
    avg_n5 = sum(r["ndcg@5"] for r in results) / len(results)
    avg_n10 = sum(r["ndcg@10"] for r in results) / len(results)

    print(f"\nOverall ({len(results)} questions):")
    print(f"  R@5:    {avg_r5:.1%} ({sum(r['recall@5'] for r in results):.0f}/{len(results)})")
    print(f"  R@10:   {avg_r10:.1%}")
    print(f"  NDCG@5: {avg_n5:.3f}")
    print(f"  NDCG@10:{avg_n10:.3f}")
    print(f"  Time:   {total_time:.1f}s total, {total_time/len(results):.2f}s/q avg")

    print(f"\nPer question type:")
    for qtype, qresults in sorted(type_results.items()):
        qr5 = sum(r["recall@5"] for r in qresults) / len(qresults)
        qr10 = sum(r["recall@10"] for r in qresults) / len(qresults)
        print(f"  {qtype:30s} R@5={qr5:.1%} R@10={qr10:.1%} (n={len(qresults)})")

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"results_engram_{mode}_{embed_model}_{timestamp}.json"
    out_path = Path(__file__).parent / out_name

    output = {
        "mode": mode,
        "embed_model": embed_model,
        "include_assistant": include_assistant,
        "use_prefs": use_prefs,
        "use_reranker": use_reranker or mode in ("rerank", "full"),
        "n_questions": len(results),
        "recall@5": round(avg_r5, 4),
        "recall@10": round(avg_r10, 4),
        "ndcg@5": round(avg_n5, 4),
        "ndcg@10": round(avg_n10, 4),
        "total_time_s": round(total_time, 1),
        "per_question": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Comparison ---
    print(f"\n{'=' * 60}")
    print("COMPARISON vs MemPalace")
    print(f"{'=' * 60}")
    print(f"  MemPalace raw (all-MiniLM-L6-v2):     96.6% R@5")
    print(f"  MemPalace hybrid v4 held-out:          98.4% R@5")
    print(f"  Engram {mode} ({embed_model}):  {avg_r5:.1%} R@5")
    delta = avg_r5 - 0.966
    print(f"  Delta vs raw baseline:                 {delta:+.1%}")


def main():
    parser = argparse.ArgumentParser(description="Engram LongMemEval benchmark")
    parser.add_argument("data_path", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rerank", "full"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--embed-model",
        default="bge-large",
        help="Embedding model (default: bge-large)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for evaluation")
    parser.add_argument(
        "--no-assistant",
        action="store_true",
        help="Only index user turns (like MemPalace)",
    )
    parser.add_argument(
        "--no-prefs",
        action="store_true",
        help="Disable preference document generation",
    )
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data_path,
        mode=args.mode,
        embed_model=args.embed_model,
        limit=args.limit,
        top_k=args.top_k,
        include_assistant=not args.no_assistant,
        use_prefs=not args.no_prefs,
        use_reranker=(args.mode in ("rerank", "full")),
    )


if __name__ == "__main__":
    main()
