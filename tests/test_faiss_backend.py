"""Tests for FAISS + SQLite backend."""

import numpy as np
import pytest

from engram.backends.base import Document
from engram.backends.faiss_backend import FaissBackend


def _make_doc(doc_id: str, text: str, dim: int = 8) -> Document:
    """Create a document with a random normalized embedding."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return Document(id=doc_id, text=text, embedding=vec.tolist(), metadata={"type": "test"})


class TestFaissBackend:
    def test_add_and_count(self):
        backend = FaissBackend(dimension=8)
        docs = [_make_doc(f"d{i}", f"doc {i}") for i in range(5)]
        backend.add(docs)
        assert backend.count() == 5

    def test_add_empty(self):
        backend = FaissBackend(dimension=8)
        backend.add([])
        assert backend.count() == 0

    def test_query_returns_results(self):
        backend = FaissBackend(dimension=8)
        docs = [_make_doc(f"d{i}", f"document {i}") for i in range(10)]
        backend.add(docs)

        query_vec = np.random.randn(8).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        results = backend.query(query_vec.tolist(), top_k=3)
        assert len(results) == 3
        # Results should have scores
        assert all(r.score != 0.0 for r in results)

    def test_query_empty_index(self):
        backend = FaissBackend(dimension=8)
        query_vec = np.random.randn(8).astype(np.float32).tolist()
        results = backend.query(query_vec, top_k=5)
        assert results == []

    def test_query_similarity_order(self):
        backend = FaissBackend(dimension=8)
        # Create a known vector and a similar/dissimilar pair
        target = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        similar = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        similar = similar / np.linalg.norm(similar)
        dissimilar = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

        backend.add(
            [
                Document(id="similar", text="similar", embedding=similar.tolist()),
                Document(id="dissimilar", text="dissimilar", embedding=dissimilar.tolist()),
            ]
        )

        results = backend.query(target.tolist(), top_k=2)
        assert results[0].id == "similar"
        assert results[1].id == "dissimilar"

    def test_metadata_filter(self):
        backend = FaissBackend(dimension=8)
        vec = np.random.randn(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        backend.add(
            [
                Document(id="d1", text="a", embedding=vec.tolist(), metadata={"type": "session"}),
                Document(
                    id="d2", text="b", embedding=vec.tolist(), metadata={"type": "preference"}
                ),
            ]
        )

        results = backend.query(vec.tolist(), top_k=10, metadata_filter={"type": "session"})
        assert len(results) == 1
        assert results[0].id == "d1"

    def test_delete(self):
        backend = FaissBackend(dimension=8)
        docs = [_make_doc(f"d{i}", f"doc {i}") for i in range(3)]
        backend.add(docs)
        assert backend.count() == 3

        backend.delete(["d0", "d1"])
        assert backend.count() == 1

    def test_clear(self):
        backend = FaissBackend(dimension=8)
        docs = [_make_doc(f"d{i}", f"doc {i}") for i in range(5)]
        backend.add(docs)
        assert backend.count() == 5

        backend.clear()
        assert backend.count() == 0

    def test_persistence(self, tmp_path):
        # Write
        backend = FaissBackend(path=tmp_path / "store", dimension=8)
        docs = [_make_doc(f"d{i}", f"doc {i}") for i in range(3)]
        backend.add(docs)
        assert backend.count() == 3

        # Reload
        backend2 = FaissBackend(path=tmp_path / "store", dimension=8)
        assert backend2.count() == 3

    def test_no_embedding_raises(self):
        backend = FaissBackend(dimension=8)
        doc = Document(id="bad", text="no embedding")
        with pytest.raises(ValueError):
            backend.add([doc])
