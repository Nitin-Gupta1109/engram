"""Engram — Interactive Memory Demo

Run: streamlit run app.py
"""

import json
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Engram — Conversational Memory",
    page_icon="🧠",
    layout="wide",
)


# ── Session state init ──────────────────────────────────────────────────────

if "backend" not in st.session_state:
    st.session_state.backend = None
    st.session_state.embedder = None
    st.session_state.documents = []
    st.session_state.conversations = []
    st.session_state.store_path = None
    st.session_state.initialized = False


@st.cache_resource(show_spinner="Loading embedding model (first time takes ~1 min)...")
def load_engine():
    from engram.backends.faiss_backend import FaissBackend
    from engram.retrieval.embedder import Embedder

    store_path = Path(tempfile.mkdtemp(prefix="engram_demo_"))
    backend = FaissBackend(path=store_path, dimension=1024)
    embedder = Embedder("bge-large")
    # Warm up the model
    embedder.encode_query("warmup")
    return backend, embedder, store_path


def init_engine():
    backend, embedder, store_path = load_engine()
    st.session_state.backend = backend
    st.session_state.embedder = embedder
    st.session_state.store_path = store_path
    st.session_state.initialized = True


def ingest_conversations(conversations):
    from engram.backends.base import Document
    from engram.ingestion.parser import session_to_documents

    backend = st.session_state.backend
    embedder = st.session_state.embedder

    all_docs = []
    for conv in conversations:
        turns = conv.get("turns", [])
        parsed = session_to_documents(
            session=turns,
            session_id=conv.get("id", "session"),
            timestamp=conv.get("timestamp", ""),
            include_assistant=True,
        )
        all_docs.extend(parsed)

    if not all_docs:
        return 0

    texts = [d["text"] for d in all_docs]
    embeddings = embedder.encode_documents(texts)

    documents = []
    for i, doc_info in enumerate(all_docs):
        doc = Document(
            id=doc_info["id"],
            text=doc_info["text"],
            embedding=embeddings[i].tolist(),
            metadata=doc_info["metadata"],
        )
        documents.append(doc)

    backend.add(documents)
    st.session_state.documents.extend(documents)
    st.session_state.conversations.extend(conversations)
    return len(documents)


def search(query, top_k=5, use_hybrid=True, min_score=0.0):
    from engram.retrieval.sparse import BM25
    from engram.retrieval.pipeline import reciprocal_rank_fusion

    backend = st.session_state.backend
    embedder = st.session_state.embedder
    documents = st.session_state.documents

    query_vec = embedder.encode_query(query)
    dense_results = backend.query(
        embedding=query_vec.tolist(), top_k=top_k * 3, min_score=min_score
    )

    if not use_hybrid or not documents:
        return dense_results[:top_k]

    # Hybrid: dense + BM25 + RRF
    dense_ranking = [(d.id, d.score) for d in dense_results]

    bm25 = BM25()
    all_texts = [d.text for d in documents]
    all_ids = [d.id for d in documents]
    bm25_scores = bm25.score_query_against_docs(query, all_texts)
    sparse_ranking = sorted(zip(all_ids, bm25_scores), key=lambda x: x[1], reverse=True)

    fused = reciprocal_rank_fusion(dense_ranking, sparse_ranking)

    id_to_doc = {d.id: d for d in documents}
    results = []
    for doc_id, score in fused[:top_k]:
        if doc_id in id_to_doc:
            doc = id_to_doc[doc_id]
            doc.score = score
            results.append(doc)
    return results


# ── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_CONVERSATIONS = [
    {
        "id": "chat_001",
        "timestamp": "2024-03-10",
        "turns": [
            {"role": "user", "content": "I've been learning Python for about 3 months now. I really enjoy working with pandas for data analysis."},
            {"role": "assistant", "content": "That's great progress! Pandas is an excellent library for data manipulation."},
            {"role": "user", "content": "Yes! I made some scatter plots last week to analyze our sales data at work."},
            {"role": "assistant", "content": "Nice! Scatter plots are perfect for finding correlations in sales data."},
        ],
    },
    {
        "id": "chat_002",
        "timestamp": "2024-03-12",
        "turns": [
            {"role": "user", "content": "My dog Max has been acting weird lately. He's not eating his food."},
            {"role": "assistant", "content": "I'm sorry to hear that. Has Max had any changes in routine recently?"},
            {"role": "user", "content": "We moved to a new apartment last week. Maybe that's stressing him out."},
            {"role": "assistant", "content": "That's very likely. Dogs can be sensitive to environment changes. Give him a few days to adjust."},
        ],
    },
    {
        "id": "chat_003",
        "timestamp": "2024-03-15",
        "turns": [
            {"role": "user", "content": "I'm thinking about switching from JavaScript to TypeScript for our company's frontend."},
            {"role": "assistant", "content": "TypeScript is a great choice. The type system catches many bugs at compile time."},
            {"role": "user", "content": "Our team has 5 developers and we're using React with Next.js."},
            {"role": "assistant", "content": "Next.js has excellent TypeScript support built in. Migration can be incremental."},
        ],
    },
    {
        "id": "chat_004",
        "timestamp": "2024-03-18",
        "turns": [
            {"role": "user", "content": "I started a new job at a fintech startup called PayFlow. We use Go for microservices and Kafka for event streaming."},
            {"role": "assistant", "content": "Congratulations! Go is excellent for high-throughput systems."},
            {"role": "user", "content": "I'm working on the payments processing pipeline. Having some issues with Kafka consumer lag during peak hours."},
            {"role": "assistant", "content": "Consumer lag usually means processing can't keep up. Consider scaling consumer count to match partition count."},
        ],
    },
    {
        "id": "chat_005",
        "timestamp": "2024-03-22",
        "turns": [
            {"role": "user", "content": "My wife Emma and I are planning a trip to Japan in May. We want to visit Tokyo and Kyoto."},
            {"role": "assistant", "content": "Japan in May is beautiful! The weather is mild and pleasant."},
            {"role": "user", "content": "Emma is vegetarian so we need to find good restaurants. She also loves traditional tea ceremonies."},
            {"role": "assistant", "content": "Kyoto has excellent vegetarian options at temple restaurants. For tea ceremonies, try Urasenke Chado Research Center."},
        ],
    },
    {
        "id": "chat_006",
        "timestamp": "2024-03-25",
        "turns": [
            {"role": "user", "content": "I'm training for a half marathon in June. My current best 5K is 24 minutes."},
            {"role": "assistant", "content": "A 24-minute 5K is a solid base for half marathon prep!"},
            {"role": "user", "content": "I run 3 times a week — Tuesday, Thursday, and a long run on Sunday."},
            {"role": "assistant", "content": "Great schedule. Increase long run distance by no more than 10% per week to avoid injury."},
        ],
    },
]

SAMPLE_QUERIES = [
    "What programming languages does the user know?",
    "Tell me about the user's pet",
    "What trip is being planned?",
    "What does Emma like?",
    "What tech stack does the team use?",
    "What exercise routine do they follow?",
    "Where does the user work?",
    "What issues are they having at work?",
]


# ── UI ───────────────────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0;'>
        <h1>🧠 Engram</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            High-recall conversational memory retrieval — 98.4% R@5 on LongMemEval
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize engine
if not st.session_state.initialized:
    init_engine()

# Tabs
tab_demo, tab_custom, tab_about = st.tabs(["🎮 Try It", "📝 Your Own Data", "ℹ️ About"])


# ── Tab 1: Demo ──────────────────────────────────────────────────────────────

with tab_demo:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Conversations in Memory")

        # Load sample data button
        if st.button("Load Sample Conversations", type="primary", use_container_width=True):
            with st.spinner("Embedding conversations..."):
                count = ingest_conversations(SAMPLE_CONVERSATIONS)
                st.success(f"Loaded {len(SAMPLE_CONVERSATIONS)} conversations ({count} documents)")

        # Show loaded conversations
        if st.session_state.conversations:
            st.caption(
                f"{len(st.session_state.conversations)} conversations | "
                f"{len(st.session_state.documents)} documents in memory"
            )
            for conv in st.session_state.conversations:
                with st.expander(f"🗣️ {conv['id']} — {conv.get('timestamp', '')}"):
                    for turn in conv.get("turns", []):
                        role = turn["role"]
                        icon = "👤" if role == "user" else "🤖"
                        st.markdown(f"**{icon} {role.title()}**: {turn['content']}")
        else:
            st.info("Click 'Load Sample Conversations' to get started!")

    with col_right:
        st.subheader("Search Memory")

        # Search mode
        use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=True)
        scol1, scol2 = st.columns(2)
        top_k = scol1.slider("Results to show", 1, 10, 3)
        min_score = scol2.slider("Min relevance score", 0.0, 1.0, 0.45, 0.05)

        # Quick queries
        st.caption("Try a sample query:")
        query_cols = st.columns(2)
        selected_query = None
        for i, q in enumerate(SAMPLE_QUERIES):
            col = query_cols[i % 2]
            if col.button(q, key=f"sample_{i}", use_container_width=True):
                selected_query = q

        # Custom query
        query = st.text_input(
            "Or type your own query:",
            value=selected_query or "",
            placeholder="What do you want to remember?",
        )

        if query and st.session_state.backend and st.session_state.backend.count() > 0:
            with st.spinner("Searching..."):
                results = search(query, top_k=top_k, use_hybrid=use_hybrid, min_score=min_score)

            if results:
                for i, doc in enumerate(results):
                    meta = doc.metadata or {}
                    score_pct = min(doc.score * 100, 100) if doc.score < 1 else doc.score * 100
                    doc_type = meta.get("type", "session")

                    with st.container():
                        type_badge = {
                            "session": "📄",
                            "preference": "❤️",
                            "topic": "🏷️",
                            "assistant": "🤖",
                        }.get(doc_type, "📄")

                        st.markdown(
                            f"**{type_badge} Result {i+1}** — "
                            f"`{meta.get('session_id', '?')}` | "
                            f"`{meta.get('timestamp', '')}` | "
                            f"Score: `{doc.score:.4f}` | "
                            f"Type: `{doc_type}`"
                        )
                        st.markdown(f"> {doc.text[:300]}{'...' if len(doc.text) > 300 else ''}")
                        st.divider()
            else:
                st.info(
                    f"No relevant results found (min score: {min_score}). "
                    "Try lowering the relevance threshold or rephrasing your query."
                )

        elif query:
            st.warning("Load some conversations first!")


# ── Tab 2: Custom Data ──────────────────────────────────────────────────────

with tab_custom:
    st.subheader("Ingest Your Own Conversations")

    st.markdown(
        """
        Upload a JSON file or paste conversations below. Format:
        ```json
        [
          {
            "id": "session_001",
            "timestamp": "2024-01-15",
            "turns": [
              {"role": "user", "content": "Hello!"},
              {"role": "assistant", "content": "Hi there!"}
            ]
          }
        ]
        ```
        """
    )

    upload_col, paste_col = st.columns(2)

    with upload_col:
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                if isinstance(data, list):
                    with st.spinner("Embedding..."):
                        count = ingest_conversations(data)
                    st.success(f"Ingested {count} documents from {len(data)} conversations")
                else:
                    st.error("JSON must be a list of conversation objects")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

    with paste_col:
        pasted = st.text_area("Or paste JSON here:", height=200)
        if st.button("Ingest pasted data"):
            if pasted.strip():
                try:
                    data = json.loads(pasted)
                    if isinstance(data, list):
                        with st.spinner("Embedding..."):
                            count = ingest_conversations(data)
                        st.success(f"Ingested {count} documents from {len(data)} conversations")
                    else:
                        st.error("JSON must be a list of conversation objects")
                except json.JSONDecodeError:
                    st.error("Invalid JSON")

    # Manual conversation builder
    st.divider()
    st.subheader("Or Build a Conversation")

    if "manual_turns" not in st.session_state:
        st.session_state.manual_turns = []

    mcol1, mcol2 = st.columns([1, 3])
    with mcol1:
        role = st.selectbox("Role", ["user", "assistant"])
    with mcol2:
        content = st.text_input("Message", placeholder="Type a message...")

    if st.button("Add Turn") and content:
        st.session_state.manual_turns.append({"role": role, "content": content})

    if st.session_state.manual_turns:
        st.caption("Current conversation:")
        for t in st.session_state.manual_turns:
            icon = "👤" if t["role"] == "user" else "🤖"
            st.markdown(f"{icon} **{t['role']}**: {t['content']}")

        bcol1, bcol2 = st.columns(2)
        if bcol1.button("Save to Memory", type="primary"):
            conv = {
                "id": f"manual_{len(st.session_state.conversations) + 1}",
                "timestamp": "2024-01-01",
                "turns": st.session_state.manual_turns,
            }
            with st.spinner("Embedding..."):
                count = ingest_conversations([conv])
            st.success(f"Saved! ({count} documents)")
            st.session_state.manual_turns = []

        if bcol2.button("Clear"):
            st.session_state.manual_turns = []


# ── Tab 3: About ────────────────────────────────────────────────────────────

with tab_about:
    st.subheader("How Engram Works")

    st.markdown(
        """
        ### Three-Stage Retrieval Pipeline

        ```
        Query → Dense Search (bge-large) → BM25 Keyword Search → RRF Fusion → Results
        ```

        1. **Dense retrieval** — bge-large-en-v1.5 (1024-dim) encodes your query and finds
           semantically similar documents via cosine similarity
        2. **Sparse retrieval** — BM25 catches exact keyword matches that embeddings might miss
        3. **RRF fusion** — Reciprocal Rank Fusion combines both rankings without needing
           score calibration

        ### Document Types

        When you ingest a conversation, Engram creates multiple searchable documents:

        | Type | Purpose |
        |------|---------|
        | 📄 **Session** | Full conversation text |
        | ❤️ **Preference** | Extracted user preferences and habits |
        | 🏷️ **Topic** | Key nouns for vocabulary bridging |
        | 🤖 **Assistant** | What the AI said (separate from user turns) |

        ### Benchmark Results (LongMemEval)

        | Metric | Score |
        |--------|-------|
        | R@5 | **98.4%** (492/500) |
        | R@10 | 99.4% |
        | NDCG@5 | 0.934 |

        ### Links

        - [GitHub](https://github.com/Nitin-Gupta1109/engram)
        - [PyPI](https://pypi.org/project/engram-search/)
        - Install: `pip install engram-search`
        """
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Memory Stats")
    if st.session_state.backend:
        st.metric("Documents", st.session_state.backend.count())
        st.metric("Conversations", len(st.session_state.conversations))

    st.divider()
    st.markdown(
        """
        **Engram** v0.1.0

        [GitHub](https://github.com/Nitin-Gupta1109/engram) ·
        [PyPI](https://pypi.org/project/engram-search/)

        `pip install engram-search`
        """
    )

    if st.button("Reset Memory"):
        if st.session_state.backend:
            st.session_state.backend.clear()
            st.session_state.documents = []
            st.session_state.conversations = []
            st.rerun()
