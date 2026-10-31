"""Tests for slm-training/rag.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from rag import (  # noqa: E402
    Document,
    RetrievalResult,
    TFIDFIndex,
    augmented_generation,
    build_index,
    chunk_document,
)


class TestChunkDocument:
    def test_empty_text_returns_empty(self) -> None:
        assert chunk_document("") == []

    def test_short_text_returns_one_chunk(self) -> None:
        result = chunk_document("hello world", chunk_size=10)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_splits_into_multiple_chunks(self) -> None:
        words = " ".join(f"w{i}" for i in range(30))
        result = chunk_document(words, chunk_size=10, overlap=2)
        assert len(result) > 1

    def test_overlap_prevents_boundary_loss(self) -> None:
        words = " ".join(f"word{i}" for i in range(25))
        result = chunk_document(words, chunk_size=10, overlap=5)
        first = result[0].text.split()
        second = result[1].text.split()
        # Last 5 words of first chunk should appear in second chunk
        overlap_words = first[-5:]
        assert all(w in second for w in overlap_words)

    def test_source_is_propagated(self) -> None:
        result = chunk_document("hello world", source="my_doc")
        assert result[0].source == "my_doc"

    def test_chunk_ids_are_sequential(self) -> None:
        words = " ".join(f"w{i}" for i in range(25))
        result = chunk_document(words, chunk_size=10, overlap=2)
        assert [c.chunk_id for c in result] == list(range(len(result)))


class TestTFIDFIndex:
    def test_empty_fit(self) -> None:
        index = TFIDFIndex()
        index.fit([])
        assert index.query("anything") == []

    def test_query_returns_top_k(self) -> None:
        docs = [
            Document(text="cats are cute pets", source="a"),
            Document(text="dogs are friendly pets", source="b"),
            Document(text="cars have engines", source="c"),
        ]
        index = TFIDFIndex()
        index.fit(docs)
        results = index.query("cute cats", top_k=2)
        assert len(results) == 2

    def test_query_ranks_relevant_document_higher(self) -> None:
        docs = [
            Document(text="python programming language", source="code"),
            Document(text="cooking pasta recipe", source="food"),
            Document(text="javascript web framework", source="web"),
        ]
        index = TFIDFIndex()
        index.fit(docs)
        results = index.query("python programming", top_k=3)
        assert results[0].document.source == "code"
        assert results[0].score > results[1].score

    def test_empty_query_returns_zero_scores(self) -> None:
        docs = [Document(text="some text here")]
        index = TFIDFIndex()
        index.fit(docs)
        results = index.query("")
        assert len(results) == len(docs)
        assert all(r.score == 0.0 for r in results)

    def test_fit_builds_vocabulary(self) -> None:
        docs = [Document(text="alpha beta"), Document(text="beta gamma")]
        index = TFIDFIndex()
        index.fit(docs)
        assert "alpha" in index.vocabulary
        assert "beta" in index.vocabulary
        assert "gamma" in index.vocabulary

    def test_idf_penalises_common_terms(self) -> None:
        docs = [
            Document(text="data data data"),
            Document(text="data model"),
            Document(text="model"),
            Document(text="python"),
        ]
        index = TFIDFIndex()
        index.fit(docs)
        # "data" appears in 2 docs, "model" in 2 docs, "python" in 1 doc.
        assert index._idf["data"] == index._idf["model"]
        assert index._idf["python"] > index._idf["data"]


class TestCosineSimilarity:
    def test_identical_vectors_have_score_one(self) -> None:
        index = TFIDFIndex()
        vec = {"a": 1.0, "b": 0.5}
        assert index._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_score_zero(self) -> None:
        index = TFIDFIndex()
        assert index._cosine_similarity({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self) -> None:
        index = TFIDFIndex()
        assert index._cosine_similarity({}, {"a": 1.0}) == pytest.approx(0.0)
        assert index._cosine_similarity({"a": 1.0}, {}) == pytest.approx(0.0)


class TestBuildIndex:
    def test_with_text(self) -> None:
        index = build_index(text="hello world test")
        assert len(index.documents) > 0

    def test_with_documents(self) -> None:
        docs = [Document(text="foo bar")]
        index = build_index(documents=docs)
        assert index.documents == docs

    def test_empty_text_returns_empty_index(self) -> None:
        index = build_index(text="")
        assert index.query("anything") == []


class TestAugmentedGeneration:
    def test_no_results_returns_fallback(self) -> None:
        prompt = augmented_generation("what is up?", [])
        assert "Question: what is up?" in prompt

    def test_includes_context(self) -> None:
        result = RetrievalResult(
            document=Document(text="The sky is blue", source="science"),
            score=0.9,
        )
        prompt = augmented_generation("What color is the sky?", [result])
        assert "The sky is blue" in prompt
        assert "What color is the sky?" in prompt

    def test_respects_max_context_chars(self) -> None:
        docs = [
            RetrievalResult(
                document=Document(text="A" * 500, source="a"),
                score=0.9,
            ),
            RetrievalResult(
                document=Document(text="B" * 500, source="b"),
                score=0.8,
            ),
        ]
        prompt = augmented_generation("q?", docs, max_context_chars=600)
        assert len(prompt) < 1000

    def test_single_result_formats_correctly(self) -> None:
        result = RetrievalResult(
            document=Document(text="context text", source="src"),
            score=1.0,
        )
        prompt = augmented_generation("query", [result])
        assert prompt.startswith("Context:\n")
        assert "[src]" in prompt
