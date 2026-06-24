"""
Part 6, File 3: Late Chunking — Preserving Cross-Chunk Context

Two chunking strategies compared:

  1. Early Chunking (standard):
       Split text → embed each chunk independently.
       Problem: chunks at boundaries lose context from neighbouring text.

  2. Late Chunking:
       Embed the text using *overlapping windows* that capture surrounding context,
       then extract the per-chunk embedding from the windowed representation.
       This preserves cross-chunk relationships.

In production, true "late chunking" is done with models that output per-token
embeddings (e.g. Jina's late-chunking or ColBERT). Here we simulate the concept
using overlapping context windows with OpenAI embeddings to demonstrate the
principle and measure its impact.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain v1 imports
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "part6_03_late_chunking"

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


# ═══════════════════════════════════════════════════════════════════════════
# Utility: cosine similarity
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


# ═══════════════════════════════════════════════════════════════════════════
# LateChunkingPipeline
# ═══════════════════════════════════════════════════════════════════════════

class LateChunkingPipeline:
    """
    Compare early (standard) chunking against late chunking that embeds
    overlapping context windows, preserving cross-chunk understanding.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        context_window_multiplier: int = 3,
    ) -> None:
        """
        Args:
            embedding_model: OpenAI embedding model name.
            chunk_size: target characters per chunk.
            chunk_overlap: overlap between adjacent chunks.
            context_window_multiplier: for late chunking, each chunk is
                embedded with this many times the chunk_size of surrounding
                context. E.g. 3 means the window = 3 × chunk_size centred
                on the chunk.
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ctx_mult = context_window_multiplier

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Step 1: split text into chunks (shared by both approaches)
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunks using the configured splitter."""
        docs = self._splitter.create_documents([text])
        return [d.page_content for d in docs]

    # ------------------------------------------------------------------
    # Early chunking: embed each chunk independently
    # ------------------------------------------------------------------

    def early_chunk_embed(self, text: str, chunk_size: int | None = None) -> list[list[float]]:
        """
        Standard (early) chunking: split text, embed each chunk independently.

        Returns a list of embedding vectors, one per chunk.
        """
        if chunk_size:
            old = self._splitter._chunk_size
            self._splitter._chunk_size = chunk_size
            chunks = self.split_text(text)
            self._splitter._chunk_size = old
        else:
            chunks = self.split_text(text)

        print(f"    Early chunking: {len(chunks)} chunks, embedding independently …")
        embeddings = self.embeddings.embed_documents(chunks)
        return embeddings

    # ------------------------------------------------------------------
    # Late chunking: embed with overlapping context window
    # ------------------------------------------------------------------

    def late_chunk_embed(self, text: str, chunk_size: int | None = None) -> list[list[float]]:
        """
        Late chunking: for each chunk, build a *context window* that includes
        surrounding text (up to context_window_multiplier × chunk_size), embed
        that window, and use its embedding as the chunk's representation.

        This means the embedding captures not just the chunk's own content, but
        also the neighbouring context — preserving cross-chunk relationships.

        Returns a list of embedding vectors, one per chunk.
        """
        if chunk_size:
            old = self._splitter._chunk_size
            self._splitter._chunk_size = chunk_size
            chunks = self.split_text(text)
            self._splitter._chunk_size = old
        else:
            chunks = self.split_text(text)

        context_radius = (self.ctx_mult * self.chunk_size) // 2

        windowed_texts: list[str] = []

        for chunk in chunks:
            # Find where this chunk appears in the original text
            start_idx = text.find(chunk)
            if start_idx == -1:
                # Fallback: just use the chunk itself
                windowed_texts.append(chunk)
                continue

            end_idx = start_idx + len(chunk)

            # Expand to context window
            window_start = max(0, start_idx - context_radius)
            window_end = min(len(text), end_idx + context_radius)
            window_text = text[window_start:window_end]

            windowed_texts.append(window_text)

        print(f"    Late chunking: {len(chunks)} chunks, embedding with "
              f"context window (radius={context_radius} chars) …")

        embeddings = self.embeddings.embed_documents(windowed_texts)
        return embeddings

    # ------------------------------------------------------------------
    # Retrieve: build a Chroma store and search
    # ------------------------------------------------------------------

    def _build_and_search(
        self,
        text: str,
        query: str,
        method: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Build a vectorstore using the specified method, then search.

        Returns list of (chunk_text, similarity_score) tuples.
        """
        chunks = self.split_text(text)

        if method == "early":
            embeddings = self.early_chunk_embed(text)
        else:
            embeddings = self.late_chunk_embed(text)

        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Compute similarities
        scored: list[tuple[str, float]] = []
        for chunk_text, chunk_emb in zip(chunks, embeddings):
            sim = cosine_similarity(query_embedding, chunk_emb)
            scored.append((chunk_text, sim))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Compare both methods on a query
    # ------------------------------------------------------------------

    def compare(
        self,
        text: str,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Run both early and late chunking retrieval on the same query.

        Returns comparison dict with results and similarity scores.
        """
        print(f"\n  🔍 Query: \"{query}\"")

        print("\n  ── Early Chunking ──")
        early_results = self._build_and_search(text, query, "early", top_k)

        print("\n  ── Late Chunking ──")
        late_results = self._build_and_search(text, query, "late", top_k)

        return {
            "query": query,
            "early": early_results,
            "late": late_results,
        }

    # ------------------------------------------------------------------
    # Pretty-print comparison
    # ------------------------------------------------------------------

    @staticmethod
    def print_comparison(comparison: dict[str, Any]) -> None:
        """Print a formatted comparison of early vs. late chunking results."""
        print(f"\n  {'─' * 70}")
        print(f"  Query: \"{comparison['query']}\"")
        print(f"  {'─' * 70}")

        for method_name, results in [("Early Chunking", comparison["early"]),
                                      ("Late Chunking", comparison["late"])]:
            print(f"\n  📌 {method_name}:")
            for rank, (chunk, score) in enumerate(results, 1):
                snippet = chunk[:120].replace("\n", " ").strip()
                if len(chunk) > 120:
                    snippet += "…"
                print(f"    {rank}. [sim={score:.4f}] {snippet}")

        # Compare top scores
        early_top = comparison["early"][0][1] if comparison["early"] else 0
        late_top = comparison["late"][0][1] if comparison["late"] else 0
        diff = late_top - early_top

        print(f"\n  📊 Top-1 similarity — Early: {early_top:.4f}, Late: {late_top:.4f} "
              f"(diff: {diff:+.4f})")

        if diff > 0.005:
            print("  ✅ Late chunking produced a more relevant top result!")
        elif diff < -0.005:
            print("  ℹ️  Early chunking scored higher (query may not need cross-chunk context)")
        else:
            print("  ➡️  Roughly equivalent for this query")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Demonstrate early vs. late chunking and compare context preservation."""
    print("=" * 70)
    print("  Part 6.3 — Late Chunking: Cross-Chunk Context Preservation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Validate environment
    # ------------------------------------------------------------------
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not found in environment.")
        print("   Copy .env.example → .env and add your key.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load documents
    # ------------------------------------------------------------------
    print(f"\n📂 Loading documents from {DOCS_DIR} …")
    all_text_parts: list[str] = []
    for fpath in sorted(DOCS_DIR.iterdir()):
        if fpath.suffix in (".txt", ".md"):
            try:
                content = fpath.read_text(encoding="utf-8")
                all_text_parts.append(content)
                print(f"  📄 Loaded {fpath.name} ({len(content):,} chars)")
            except Exception as exc:
                print(f"  ⚠️  Skipped {fpath.name}: {exc}")

    full_text = "\n\n" + ("=" * 40) + "\n\n".join(all_text_parts)
    print(f"  ✅ Combined text: {len(full_text):,} characters")

    # ------------------------------------------------------------------
    # 3. Instantiate pipeline
    # ------------------------------------------------------------------
    pipeline = LateChunkingPipeline(
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=300,
        chunk_overlap=50,
        context_window_multiplier=3,
    )

    # ------------------------------------------------------------------
    # 4. Show chunk counts
    # ------------------------------------------------------------------
    chunks = pipeline.split_text(full_text)
    print(f"\n  🔪 Total chunks: {len(chunks)}")
    print(f"  📏 Avg chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    # ------------------------------------------------------------------
    # 5. Run comparisons — especially queries that require cross-chunk context
    # ------------------------------------------------------------------
    queries = [
        # This query requires understanding that "the policy" refers to ACME Corp's
        # leave policy — context that may be in a different chunk.
        "What is the carryover limit for unused vacation days at ACME?",

        # This query asks about a comparison that spans multiple chunks.
        "How does HNSW compare to IVFFlat for vector search?",

        # Cross-document query: connects employee policy with tech docs.
        "What are the best practices for production deployment?",
    ]

    print("\n🏁 Running early vs. late chunking comparison …")
    all_comparisons: list[dict[str, Any]] = []
    for query in queries:
        comp = pipeline.compare(full_text, query, top_k=3)
        all_comparisons.append(comp)

    # ------------------------------------------------------------------
    # 6. Print all comparisons
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📊 Results: Early Chunking vs. Late Chunking")
    print("=" * 70)

    for comp in all_comparisons:
        LateChunkingPipeline.print_comparison(comp)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📋 Summary")
    print("=" * 70)

    improvements = 0
    for comp in all_comparisons:
        early_top = comp["early"][0][1] if comp["early"] else 0
        late_top = comp["late"][0][1] if comp["late"] else 0
        if late_top > early_top + 0.005:
            improvements += 1

    print(f"\n  Queries tested:            {len(queries)}")
    print(f"  Late chunking improved:    {improvements}/{len(queries)}")
    print()
    print("  Key Takeaways:")
    print("  • Late chunking embeds each chunk with surrounding context")
    print("  • This is most helpful for queries that require cross-chunk understanding")
    print("  • The trade-off is slightly higher embedding cost (larger text windows)")
    print("  • In production, use models with native per-token embeddings (e.g. Jina)")
    print("    for true late chunking without the extra API calls")

    print("\n✅ Late chunking comparison complete!")


if __name__ == "__main__":
    main()
