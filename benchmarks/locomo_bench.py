#!/usr/bin/env python3
"""
Engram x LoCoMo Benchmark
===========================

Evaluates Engram's retrieval against the LoCoMo benchmark (Snap Research).
https://github.com/snap-research/locomo

LoCoMo contains 10 long conversations (19-35 sessions each, 300+ turns),
with 1986 QA pairs across 5 categories:
    1: Single-hop factual (identity/attributes)
    2: Temporal (dates, events)
    3: Multi-hop inference
    4: Contextual details / motivations
    5: Adversarial (speaker confusion)

For each conversation:
1. Ingest all sessions as Engram documents (user turns, preferences, topics)
2. For each QA, retrieve top-k and check if evidence session is found
3. Report Recall@k, NDCG@k per category and overall

Usage:
    python benchmarks/locomo_bench.py data/locomo10.json
    python benchmarks/locomo_bench.py data/locomo10.json --mode hybrid
    python benchmarks/locomo_bench.py data/locomo10.json --limit 2
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# =============================================================================
# METRICS
# =============================================================================

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "contextual",
    5: "adversarial",
}


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
# LOCOMO DATA PARSING
# =============================================================================


def parse_locomo_conversation(conv):
    """Convert LoCoMo conversation format to Engram session format.

    LoCoMo format: conversation object with session_1, session_2, etc.
    Each session has turns with {speaker, dia_id, text}.

    Returns (sessions, speaker_names) where sessions is a list of
    {session_id, timestamp, turns} and speaker_names maps role -> name.
    """
    speaker_a = conv["speaker_a"]
    speaker_b = conv.get("speaker_b", "")
    speaker_names = {"user": speaker_a, "assistant": speaker_b}
    sessions = []

    # Find all session keys
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    for sk in session_keys:
        session_num = int(sk.split("_")[1])
        dt_key = f"{sk}_date_time"
        timestamp = conv.get(dt_key, "")
        raw_turns = conv[sk]

        # Convert to Engram format: {role: "user"/"assistant", content: "..."}
        # In LoCoMo, speaker_a is treated as "user", speaker_b as "assistant"
        turns = []
        for turn in raw_turns:
            role = "user" if turn["speaker"] == speaker_a else "assistant"
            turns.append({"role": role, "content": turn["text"]})

        sessions.append({
            "session_num": session_num,
            "session_id": f"session_{session_num}",
            "timestamp": timestamp,
            "turns": turns,
        })

    return sessions, speaker_names


def evidence_to_session_ids(evidence_list):
    """Convert LoCoMo evidence IDs to session IDs.

    Evidence format: 'D{session}:{turn}' e.g. 'D1:3' -> session_1
    """
    session_ids = set()
    for ev in evidence_list:
        parts = ev.split(":")
        sess_num = parts[0].replace("D", "")
        session_ids.add(f"session_{sess_num}")
    return session_ids


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
    """Run the LoCoMo benchmark."""
    from engram.backends.base import Document
    from engram.ingestion.parser import session_to_documents
    from engram.retrieval.embedder import Embedder
    from engram.retrieval.pipeline import RetrievalPipeline, reciprocal_rank_fusion

    # Load dataset
    print(f"Loading LoCoMo dataset from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    total_convos = len(data)
    if limit > 0:
        data = data[:limit]
    total_qa = sum(len(e["qa"]) for e in data)
    print(f"Loaded {total_convos} conversations, using {len(data)} ({total_qa} QA pairs)")

    # Initialize embedder
    print(f"Initializing embedder: {embed_model}...")
    embedder = Embedder(embed_model)

    # Initialize reranker if needed
    reranker = None
    if use_reranker or mode in ("rerank", "full"):
        print("Initializing cross-encoder reranker...")
        from engram.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

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

    print(f"\nRunning LoCoMo benchmark: mode={mode}, embed={embed_model}, top_k={top_k}")
    print(f"Include assistant turns: {include_assistant}")
    print(f"Preference docs: {use_prefs}")
    print(f"Cross-encoder reranker: {use_reranker or mode in ('rerank', 'full')}")
    print("=" * 60)

    for ci, entry in enumerate(data):
        sample_id = entry.get("sample_id", f"conv_{ci}")
        conv = entry["conversation"]
        qa_list = entry["qa"]

        print(f"\n[Conversation {ci + 1}/{len(data)}] {sample_id} — {len(qa_list)} questions")

        # --- Parse and ingest all sessions for this conversation ---
        sessions, speaker_names = parse_locomo_conversation(conv)
        print(f"  Sessions: {len(sessions)}, ingesting...")

        all_docs_raw = []
        session_id_for_doc = {}  # doc_id -> session_id

        for sess in sessions:
            parsed = session_to_documents(
                session=sess["turns"],
                session_id=sess["session_id"],
                timestamp=sess["timestamp"],
                include_assistant=include_assistant,
                generate_preference_doc=use_prefs,
                speaker_names=speaker_names,
            )
            for doc_info in parsed:
                all_docs_raw.append(doc_info)
                session_id_for_doc[doc_info["id"]] = sess["session_id"]

        if not all_docs_raw:
            print("  WARNING: No documents generated, skipping")
            continue

        # Embed all documents for this conversation
        texts = [d["text"] for d in all_docs_raw]
        print(f"  Embedding {len(all_docs_raw)} documents...")
        t_embed = time.time()
        embeddings = embedder.encode_documents(texts)
        print(f"  Embedded in {time.time() - t_embed:.1f}s")

        documents = []
        for i, doc_info in enumerate(all_docs_raw):
            documents.append(
                Document(
                    id=doc_info["id"],
                    text=doc_info["text"],
                    embedding=embeddings[i].tolist(),
                    metadata=doc_info["metadata"],
                )
            )

        # --- Evaluate each QA pair ---
        conv_hits_5 = 0
        conv_total = 0

        for qi, qa in enumerate(qa_list):
            question = qa["question"]
            category = qa["category"]
            cat_name = CATEGORY_NAMES.get(category, f"cat_{category}")
            evidence = qa.get("evidence", [])

            if not evidence:
                continue  # skip QAs without evidence

            correct_session_ids = evidence_to_session_ids(evidence)

            t0 = time.time()

            # --- Retrieve ---
            if mode == "dense":
                query_vec = embedder.encode_query(question)
                scored = []
                for doc in documents:
                    emb = np.array(doc.embedding)
                    sim = float(np.dot(query_vec, emb))
                    scored.append((doc, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                retrieved = [d for d, _ in scored[: top_k * 2]]
            else:
                retrieved = pipeline.search(
                    query=question,
                    documents=documents,
                    top_k=top_k * 2,
                )

            elapsed = time.time() - t0
            total_time += elapsed

            # --- Map retrieved docs to session IDs ---
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
                "sample_id": sample_id,
                "question": question,
                "category": category,
                "category_name": cat_name,
                "recall@5": r5,
                "recall@10": r10,
                "ndcg@5": n5,
                "ndcg@10": n10,
                "time": round(elapsed, 3),
            }
            results.append(result)
            type_results[cat_name].append(result)

            if r5 > 0:
                conv_hits_5 += 1
            conv_total += 1

        if conv_total > 0:
            conv_r5 = conv_hits_5 / conv_total
            print(f"  R@5: {conv_r5:.1%} ({conv_hits_5}/{conv_total})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("LOCOMO RESULTS SUMMARY")
    print("=" * 60)

    avg_r5 = sum(r["recall@5"] for r in results) / len(results)
    avg_r10 = sum(r["recall@10"] for r in results) / len(results)
    avg_n5 = sum(r["ndcg@5"] for r in results) / len(results)
    avg_n10 = sum(r["ndcg@10"] for r in results) / len(results)

    print(f"\nOverall ({len(results)} questions):")
    print(f"  R@5:     {avg_r5:.1%} ({sum(r['recall@5'] for r in results):.0f}/{len(results)})")
    print(f"  R@10:    {avg_r10:.1%}")
    print(f"  NDCG@5:  {avg_n5:.3f}")
    print(f"  NDCG@10: {avg_n10:.3f}")
    print(f"  Time:    {total_time:.1f}s total, {total_time / len(results):.2f}s/q avg")

    print(f"\nPer category:")
    for cat_name in ["single-hop", "temporal", "multi-hop", "contextual", "adversarial"]:
        qresults = type_results.get(cat_name, [])
        if not qresults:
            continue
        qr5 = sum(r["recall@5"] for r in qresults) / len(qresults)
        qr10 = sum(r["recall@10"] for r in qresults) / len(qresults)
        qn5 = sum(r["ndcg@5"] for r in qresults) / len(qresults)
        print(
            f"  {cat_name:15s} R@5={qr5:.1%}  R@10={qr10:.1%}  "
            f"NDCG@5={qn5:.3f}  (n={len(qresults)})"
        )

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"results_locomo_{mode}_{embed_model}_{timestamp}.json"
    out_path = Path(__file__).parent / out_name

    output = {
        "benchmark": "LoCoMo",
        "mode": mode,
        "embed_model": embed_model,
        "include_assistant": include_assistant,
        "use_prefs": use_prefs,
        "use_reranker": use_reranker or mode in ("rerank", "full"),
        "n_conversations": len(data),
        "n_questions": len(results),
        "recall@5": round(avg_r5, 4),
        "recall@10": round(avg_r10, 4),
        "ndcg@5": round(avg_n5, 4),
        "ndcg@10": round(avg_n10, 4),
        "total_time_s": round(total_time, 1),
        "per_category": {
            cat_name: {
                "n": len(qresults),
                "recall@5": round(
                    sum(r["recall@5"] for r in qresults) / len(qresults), 4
                ),
                "recall@10": round(
                    sum(r["recall@10"] for r in qresults) / len(qresults), 4
                ),
            }
            for cat_name, qresults in type_results.items()
        },
        "per_question": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Engram LoCoMo benchmark")
    parser.add_argument("data_path", help="Path to locomo10.json")
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
    parser.add_argument("--limit", type=int, default=0, help="Limit to N conversations")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for evaluation")
    parser.add_argument(
        "--no-assistant",
        action="store_true",
        help="Only index user turns",
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
