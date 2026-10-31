"""CLI demo: index a sample document and answer queries with the RAG pipeline.

Usage:
    python rag_demo.py --query "What is the main topic?"
    python rag_demo.py --query "Why?" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from rag import (  # noqa: E402
    augmented_generation,
    build_index,
)

SAMPLE_TEXT = """Retrieval-Augmented Generation combines information retrieval
with text generation.  Instead of relying solely on a model's parametric
memory, RAG fetches relevant passages from an external knowledge source and
injects them into the prompt.  This grounds the answer in real evidence and
reduces hallucination.

The pipeline has three stages.  First, the corpus is split into overlapping
chunks so that no single passage is too long for the context window.  Second,
each chunk is embedded as a sparse TF-IDF vector.  Third, a query is encoded
with the same vocabulary, and cosine similarity ranks the chunks.  The top
chunks are prepended to the prompt, and the model generates an answer that
cites the retrieved evidence.

RAG was first described in the 2020 paper by Lewis et al.  It showed that
even a small model could answer open-domain questions accurately when given
the right retrieved passages.  The approach scales to large corpora because
the index is sparse and cheap to query.

Chunking strategy matters.  Overlap prevents key phrases from being split
across boundaries.  A typical setting is two-hundred words per chunk with
forty words of overlap.  Larger chunks carry more context but waste tokens;
smaller chunks are precise but miss broader meaning.

Cosine similarity measures the angle between two vectors.  It ignores the
length of the documents and focuses on how much their term distributions
overlap.  TF-IDF down-weights words that appear everywhere and highlights
words that are distinctive to a document.

Augmented generation simply concatenates the retrieved passages, the user
question, and a short instruction.  The instruction tells the model to answer
from the provided context rather than from its own training data."""


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG pipeline demo")
    ap.add_argument("--query", required=True, help="Question to answer")
    ap.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    ap.add_argument("--max-context", type=int, default=2000, help="Max context chars")
    args = ap.parse_args()

    index = build_index(text=SAMPLE_TEXT, source="sample_document")
    results = index.query(args.query, top_k=args.top_k)
    prompt = augmented_generation(
        args.query, results, max_context_chars=args.max_context
    )

    print(f"Query: {args.query}")
    print(f"Retrieved {len(results)} chunk(s)")
    for i, result in enumerate(results):
        print(
            f"  [{i}] score={result.score:.4f}  "
            f"source={result.document.source}  "
            f"chunk_id={result.document.chunk_id}"
        )
    print()
    print("--- Augmented Prompt ---")
    print(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
