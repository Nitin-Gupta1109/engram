"""Tests for retrieval pipeline utilities (no model loading required)."""

import pytest
from engram.retrieval.pipeline import (
    extract_person_names,
    extract_quoted_phrases,
    parse_temporal_offset,
    parse_date,
    reciprocal_rank_fusion,
)


class TestExtractPersonNames:
    def test_finds_names(self):
        names = extract_person_names("I talked to Sarah and John yesterday")
        assert "Sarah" in names
        assert "John" in names

    def test_filters_common_words(self):
        names = extract_person_names("What did Monday bring? The answer is here")
        assert "What" not in names
        assert "Monday" not in names
        assert "The" not in names

    def test_empty_string(self):
        assert extract_person_names("") == []

    def test_no_names(self):
        assert extract_person_names("all lowercase text here") == []


class TestExtractQuotedPhrases:
    def test_double_quotes(self):
        phrases = extract_quoted_phrases('He said "hello world" to me')
        assert "hello world" in phrases

    def test_single_quotes(self):
        phrases = extract_quoted_phrases("She said 'good morning' politely")
        assert "good morning" in phrases

    def test_short_quotes_excluded(self):
        phrases = extract_quoted_phrases('He said "hi" to me')
        assert len(phrases) == 0  # "hi" is < 3 chars

    def test_no_quotes(self):
        assert extract_quoted_phrases("no quotes here") == []


class TestParseTemporalOffset:
    def test_days_ago(self):
        result = parse_temporal_offset("What did I say 3 days ago?")
        assert result is not None
        days, tolerance = result
        assert days == 3

    def test_yesterday(self):
        result = parse_temporal_offset("What did we discuss yesterday?")
        assert result is not None
        assert result[0] == 1

    def test_week_ago(self):
        result = parse_temporal_offset("A week ago I mentioned something")
        assert result is not None
        assert result[0] == 7

    def test_month_ago(self):
        result = parse_temporal_offset("About a month ago we talked")
        assert result is not None
        assert result[0] == 30

    def test_last_year(self):
        result = parse_temporal_offset("Last year I asked about this")
        assert result is not None
        assert result[0] == 365

    def test_recently(self):
        result = parse_temporal_offset("Recently we discussed Python")
        assert result is not None
        assert result[0] == 14

    def test_no_temporal(self):
        assert parse_temporal_offset("Tell me about Python") is None

    def test_two_months_ago(self):
        result = parse_temporal_offset("Two months ago I started learning")
        assert result is not None
        assert result[0] == 60


class TestParseDate:
    def test_slash_format(self):
        dt = parse_date("2024/03/15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_dash_format(self):
        dt = parse_date("2024-03-15")
        assert dt is not None
        assert dt.year == 2024

    def test_with_extra_text(self):
        dt = parse_date("2024-03-15 (Monday)")
        assert dt is not None
        assert dt.day == 15

    def test_invalid_date(self):
        assert parse_date("not a date") is None

    def test_empty_string(self):
        assert parse_date("") is None


class TestReciprocalRankFusion:
    def test_single_ranking(self):
        ranking = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        fused = reciprocal_rank_fusion(ranking)
        # Order should be preserved
        ids = [doc_id for doc_id, _ in fused]
        assert ids == ["doc1", "doc2", "doc3"]

    def test_two_rankings_agreement(self):
        r1 = [("doc1", 0.9), ("doc2", 0.7)]
        r2 = [("doc1", 0.8), ("doc2", 0.6)]
        fused = reciprocal_rank_fusion(r1, r2)
        ids = [doc_id for doc_id, _ in fused]
        assert ids[0] == "doc1"

    def test_two_rankings_disagreement(self):
        r1 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        r2 = [("doc3", 0.9), ("doc2", 0.7), ("doc1", 0.5)]
        fused = reciprocal_rank_fusion(r1, r2)
        scores = {doc_id: s for doc_id, s in fused}
        # All docs appear at same combined rank positions, so all get equal scores
        # doc2 is rank 2 in both = 2/(k+2), others get 1/(k+1) + 1/(k+3)
        # With k=60: doc2 = 2/62, doc1 = 1/61 + 1/63, doc3 = 1/61 + 1/63
        # 1/61 + 1/63 > 2/62, so doc1 and doc3 tie and beat doc2
        assert scores["doc1"] == scores["doc3"]
        assert len(fused) == 3

    def test_disjoint_rankings(self):
        r1 = [("doc1", 0.9), ("doc2", 0.7)]
        r2 = [("doc3", 0.9), ("doc4", 0.7)]
        fused = reciprocal_rank_fusion(r1, r2)
        assert len(fused) == 4

    def test_empty_rankings(self):
        fused = reciprocal_rank_fusion()
        assert fused == []

    def test_k_parameter(self):
        r1 = [("doc1", 0.9), ("doc2", 0.7)]
        fused_default = reciprocal_rank_fusion(r1, k=60)
        fused_small_k = reciprocal_rank_fusion(r1, k=1)
        # With smaller k, the score difference between ranks is larger
        score_diff_default = fused_default[0][1] - fused_default[1][1]
        score_diff_small = fused_small_k[0][1] - fused_small_k[1][1]
        assert score_diff_small > score_diff_default

    def test_scores_are_positive(self):
        r1 = [("doc1", 0.9), ("doc2", 0.7)]
        fused = reciprocal_rank_fusion(r1)
        for _, score in fused:
            assert score > 0
