"""Tests for BM25 sparse retrieval."""

import pytest
from engram.retrieval.sparse import BM25, tokenize, STOP_WORDS


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello world, this is a test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_removes_stop_words(self):
        tokens = tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize("I a am an ok go")
        # single-char tokens should be excluded
        assert "I" not in tokens and "i" not in tokens
        assert "a" not in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_lowercases(self):
        tokens = tokenize("Python JavaScript Rust")
        assert "python" in tokens
        assert "javascript" in tokens
        assert "rust" in tokens


class TestBM25:
    def test_exact_match_scores_highest(self):
        bm25 = BM25()
        docs = [
            "the cat sat on the mat",
            "dogs play in the park",
            "birds fly in the sky",
        ]
        bm25.index(docs)
        scores = bm25.score("cat mat")
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_no_match_scores_zero(self):
        bm25 = BM25()
        docs = ["apples oranges bananas", "grapes strawberries"]
        bm25.index(docs)
        scores = bm25.score("quantum physics")
        assert all(s == 0.0 for s in scores)

    def test_empty_query(self):
        bm25 = BM25()
        bm25.index(["some document text"])
        scores = bm25.score("")
        assert scores == [0.0]

    def test_empty_corpus(self):
        bm25 = BM25()
        bm25.index([])
        scores = bm25.score("anything")
        assert scores == []

    def test_score_query_against_docs(self):
        bm25 = BM25()
        docs = ["python programming language", "java enterprise beans"]
        scores = bm25.score_query_against_docs("python programming", docs)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_tf_matters(self):
        bm25 = BM25()
        docs = [
            "python python python",
            "python java rust",
        ]
        bm25.index(docs)
        scores = bm25.score("python")
        # Higher TF should give a higher score
        assert scores[0] > scores[1]

    def test_idf_matters(self):
        bm25 = BM25()
        docs = [
            "python programming language",
            "python scripting language",
            "python data science language",
            "rust systems programming",
        ]
        bm25.index(docs)
        # "rust" has lower DF (appears in 1 doc) so higher IDF
        scores_rare = bm25.score("rust")
        scores_common = bm25.score("python")
        # The doc containing "rust" should score higher for "rust"
        # than the best "python" doc scores for "python" (since python is in 3 docs)
        assert scores_rare[3] > scores_common[0]
