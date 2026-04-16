"""Tests for session parsing and document preparation."""

import pytest
from engram.ingestion.parser import (
    extract_preferences,
    extract_topics,
    is_assistant_reference,
    session_to_documents,
)


class TestExtractPreferences:
    def test_basic_preference(self):
        turns = [{"role": "user", "content": "I prefer dark roast coffee"}]
        prefs = extract_preferences(turns)
        assert any("dark roast coffee" in p for p in prefs)

    def test_worry_pattern(self):
        turns = [{"role": "user", "content": "I'm worried about climate change"}]
        prefs = extract_preferences(turns)
        assert any("climate change" in p for p in prefs)

    def test_activity_pattern(self):
        turns = [{"role": "user", "content": "I grow tomatoes in my backyard"}]
        prefs = extract_preferences(turns)
        assert any("tomatoes" in p for p in prefs)

    def test_ignores_assistant_turns(self):
        turns = [
            {"role": "assistant", "content": "I prefer helping with Python"},
            {"role": "user", "content": "Thanks"},
        ]
        prefs = extract_preferences(turns)
        assert not any("python" in p.lower() for p in prefs)

    def test_deduplicates(self):
        turns = [
            {"role": "user", "content": "I prefer dark coffee. I prefer dark coffee."},
        ]
        prefs = extract_preferences(turns)
        coffee_prefs = [p for p in prefs if "dark coffee" in p.lower()]
        assert len(coffee_prefs) <= 1

    def test_max_15_preferences(self):
        content = ". ".join(f"I prefer option number {i:02d} here" for i in range(30))
        turns = [{"role": "user", "content": content}]
        prefs = extract_preferences(turns)
        assert len(prefs) <= 15

    def test_my_pattern(self):
        turns = [{"role": "user", "content": "my photography setup is quite elaborate"}]
        prefs = extract_preferences(turns)
        assert any("photography" in p for p in prefs)

    def test_empty_turns(self):
        assert extract_preferences([]) == []


class TestExtractTopics:
    def test_proper_nouns(self):
        turns = [{"role": "user", "content": "I visited Paris and Berlin last summer"}]
        topics = extract_topics(turns)
        assert "Paris" in topics or "Berlin" in topics

    def test_possessive_phrases(self):
        turns = [{"role": "user", "content": "my garden needs more sunlight."}]
        topics = extract_topics(turns)
        assert any("garden" in t for t in topics)

    def test_filters_stop_words(self):
        turns = [{"role": "user", "content": "What about This and That"}]
        topics = extract_topics(turns)
        assert "What" not in topics
        assert "This" not in topics
        assert "That" not in topics

    def test_max_20_topics(self):
        names = " ".join(f"Xname{i}" for i in range(30))
        turns = [{"role": "user", "content": names}]
        topics = extract_topics(turns)
        assert len(topics) <= 20

    def test_ignores_assistant_turns(self):
        turns = [
            {"role": "assistant", "content": "Let me tell you about Madagascar"},
        ]
        topics = extract_topics(turns)
        assert "Madagascar" not in topics


class TestIsAssistantReference:
    def test_positive_cases(self):
        assert is_assistant_reference("What did you suggest last time?")
        assert is_assistant_reference("You told me about a recipe")
        assert is_assistant_reference("You mentioned something about Python")
        assert is_assistant_reference("Can you remind me what you said?")
        assert is_assistant_reference("In our previous conversation you helped me")

    def test_negative_cases(self):
        assert not is_assistant_reference("What is the weather today?")
        assert not is_assistant_reference("Tell me about Python")
        assert not is_assistant_reference("How do I cook pasta?")


class TestSessionToDocuments:
    def _make_session(self):
        return [
            {"role": "user", "content": "I've been working on my garden lately"},
            {"role": "assistant", "content": "That sounds great! What are you growing?"},
            {"role": "user", "content": "I grow tomatoes and herbs"},
        ]

    def test_returns_primary_doc(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1", "2024-01-01")
        primary = [d for d in docs if d["id"] == "sess1"]
        assert len(primary) == 1
        assert "garden" in primary[0]["text"]
        assert primary[0]["is_synthetic"] is False

    def test_primary_doc_excludes_assistant_by_default(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1")
        primary = [d for d in docs if d["id"] == "sess1"][0]
        assert "That sounds great" not in primary["text"]

    def test_include_assistant_in_primary(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1", include_assistant=True)
        primary = [d for d in docs if d["id"] == "sess1"][0]
        assert "That sounds great" in primary["text"]

    def test_generates_assistant_doc(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1")
        asst_docs = [d for d in docs if d["id"] == "sess1_asst"]
        assert len(asst_docs) == 1
        assert "That sounds great" in asst_docs[0]["text"]
        assert asst_docs[0]["is_synthetic"] is True

    def test_no_assistant_doc_when_include_assistant(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1", include_assistant=True)
        asst_docs = [d for d in docs if "_asst" in d["id"]]
        assert len(asst_docs) == 0

    def test_generates_preference_doc(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1")
        pref_docs = [d for d in docs if d["id"] == "sess1_pref"]
        assert len(pref_docs) == 1
        assert "User has mentioned:" in pref_docs[0]["text"]

    def test_generates_topic_doc(self):
        session = [
            {"role": "user", "content": "I visited Paris and talked to Sarah about my garden"},
            {"role": "assistant", "content": "That sounds lovely!"},
        ]
        docs = session_to_documents(session, "sess1")
        topic_docs = [d for d in docs if d["id"] == "sess1_topic"]
        assert len(topic_docs) == 1
        assert "Topics discussed:" in topic_docs[0]["text"]

    def test_disable_synthetic_docs(self):
        session = self._make_session()
        docs = session_to_documents(
            session, "sess1",
            generate_preference_doc=False,
            generate_assistant_doc=False,
            generate_topic_doc=False,
        )
        assert all(not d["is_synthetic"] for d in docs)

    def test_metadata_includes_session_id(self):
        session = self._make_session()
        docs = session_to_documents(session, "sess1", "2024-01-01")
        for doc in docs:
            assert doc["metadata"]["session_id"] == "sess1"
            assert doc["metadata"]["timestamp"] == "2024-01-01"

    def test_empty_session(self):
        docs = session_to_documents([], "sess_empty")
        assert docs == []
