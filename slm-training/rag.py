"""Retrieval-Augmented Generation (RAG) pipeline.

Document chunking, TF-IDF embeddings, cosine-similarity retrieval, and
augmented generation.  No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single chunk of text with optional source metadata."""

    text: str
    source: str = ""
    chunk_id: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document chunk with its similarity score."""

    document: Document
    score: float


@dataclass
class TFIDFIndex:
    """In-memory TF-IDF index for sparse cosine-similarity retrieval."""

    documents: list[Document] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _term_index: dict[str, int] = field(default_factory=dict, repr=False)
    _tfidf: list[dict[str, float]] = field(default_factory=list, repr=False)

    def fit(self, documents: Sequence[Document]) -> None:
        """Build the TF-IDF index from a sequence of documents."""
        self.documents = list(documents)
        self._tfidf = []
        if not self.documents:
            self.vocabulary = []
            self._idf = {}
            self._term_index = {}
            return

        n = len(self.documents)
        doc_freqs: dict[str, int] = {}
        for doc in self.documents:
            unique_terms = set(self._tokenize(doc.text))
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        self._idf = {
            term: math.log((1 + n) / (1 + df)) + 1.0 for term, df in doc_freqs.items()
        }
        self.vocabulary = sorted(self._idf.keys())
        self._term_index = {term: i for i, term in enumerate(self.vocabulary)}

        for doc in self.documents:
            tfidf_vec: dict[str, float] = {}
            tokens = self._tokenize(doc.text)
            doc_len = len(tokens)
            term_counts: dict[str, int] = {}
            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1

            for term, count in term_counts.items():
                tf = count / doc_len if doc_len > 0 else 0.0
                tfidf_vec[term] = tf * self._idf.get(term, 0.0)

            self._tfidf.append(tfidf_vec)

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase alphanumeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _cosine_similarity(
        self, query_vec: dict[str, float], doc_vec: dict[str, float]
    ) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        if not query_vec or not doc_vec:
            return 0.0

        dot = 0.0
        for term, weight in query_vec.items():
            if term in doc_vec:
                dot += weight * doc_vec[term]

        norm_q = math.sqrt(sum(v * v for v in query_vec.values()))
        norm_d = math.sqrt(sum(v * v for v in doc_vec.values()))

        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0

        return dot / (norm_q * norm_d)

    def query(self, text: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the top-k most similar documents for the query text."""
        if not self.documents or not self._tfidf:
            return []

        tokens = self._tokenize(text)
        doc_len = len(tokens)
        term_counts: dict[str, int] = {}
        for term in tokens:
            term_counts[term] = term_counts.get(term, 0) + 1

        query_vec: dict[str, float] = {}
        for term, count in term_counts.items():
            tf = count / doc_len if doc_len > 0 else 0.0
            query_vec[term] = tf * self._idf.get(term, 0.0)

        scored = [
            RetrievalResult(
                document=doc,
                score=self._cosine_similarity(query_vec, doc_vec),
            )
            for doc, doc_vec in zip(self.documents, self._tfidf)
        ]

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def chunk_document(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
    source: str = "",
) -> list[Document]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []

    chunks: list[Document] = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(Document(text=chunk_text, source=source, chunk_id=chunk_id))
        chunk_id += 1
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_index(
    documents: Sequence[Document] | None = None,
    text: str = "",
    source: str = "",
    chunk_size: int = 200,
    overlap: int = 40,
) -> TFIDFIndex:
    """Convenience builder: chunk text and index in one step."""
    if documents is None:
        if text:
            documents = chunk_document(
                text, chunk_size=chunk_size, overlap=overlap, source=source
            )
        else:
            documents = []

    index = TFIDFIndex()
    index.fit(documents)
    return index


def augmented_generation(
    query: str,
    retrieval_results: Sequence[RetrievalResult],
    max_context_chars: int = 2000,
) -> str:
    """Build an augmented prompt from retrieved context chunks."""
    if not retrieval_results:
        return f"Question: {query}\nAnswer based on the context above."

    context_parts: list[str] = []
    total_chars = 0
    for result in retrieval_results:
        label = result.document.source or "document"
        part = f"[{label}]\n{result.document.text}"
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total_chars += len(part)

    context = "\n\n".join(context_parts)
    return (
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on the context above."
    )
