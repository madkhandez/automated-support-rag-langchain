"""
03_embeddings.py — Embedding Models for RAG

Demonstrates how text is converted into numerical vectors (embeddings)
that capture semantic meaning, enabling similarity-based retrieval:
  • OpenAI text-embedding-3-small  (1536 dimensions)
  • OpenAI text-embedding-3-large  (3072 dimensions)
  • Cosine similarity between document and query embeddings
  • Semantic similarity across different phrasing
  • Vector math: addition, subtraction, similarity operations
  • Performance benchmarking and cost estimation

Run:
    python part1_foundation/03_embeddings.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings

# ── Resolve project paths ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

# ── Pricing (per 1M tokens, as of 2024/2025) ───────────────────────
PRICING = {
    "text-embedding-3-small": 0.020,   # $0.020 per 1M tokens
    "text-embedding-3-large": 0.130,   # $0.130 per 1M tokens
}


# ════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]:
      1.0  = identical direction
      0.0  = orthogonal (unrelated)
     -1.0  = opposite direction
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    return float(np.linalg.norm(a - b))


# ════════════════════════════════════════════════════════════════════
# EmbeddingExplorer
# ════════════════════════════════════════════════════════════════════
class EmbeddingExplorer:
    """Explore OpenAI embedding models and vector operations."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file.\n"
                "Get your key at https://platform.openai.com/api-keys"
            )

        self.embeddings_small = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        self.embeddings_large = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key,
        )

        print("🔢  EmbeddingExplorer initialised")
        print("    Models: text-embedding-3-small, text-embedding-3-large")

    # ── Compare dimensions ──────────────────────────────────────────
    def compare_dimensions(self, sample_text: str = "Hello, world!") -> None:
        """Show dimension differences between small and large models.

        Args:
            sample_text: Text to embed for comparison.
        """
        print("\n" + "─" * 60)
        print("  📏  Embedding Dimensions Comparison")
        print("─" * 60)

        emb_small = self.embeddings_small.embed_query(sample_text)
        emb_large = self.embeddings_large.embed_query(sample_text)

        print(f"  Input text  : \"{sample_text}\"")
        print(f"  Small model : {len(emb_small):,} dimensions (text-embedding-3-small)")
        print(f"  Large model : {len(emb_large):,} dimensions (text-embedding-3-large)")
        print(f"  Size ratio  : {len(emb_large) / len(emb_small):.1f}x more dimensions")
        print()
        print(f"  Small first 8 values : [{', '.join(f'{v:.6f}' for v in emb_small[:8])}]")
        print(f"  Large first 8 values : [{', '.join(f'{v:.6f}' for v in emb_large[:8])}]")
        print()
        print("  💡  More dimensions = finer semantic distinctions, but higher cost")
        print("      and storage requirements. Small is usually sufficient for RAG.")

    # ── Document vs. Query similarity ───────────────────────────────
    def document_query_similarity(self) -> None:
        """Show cosine similarity between document chunks and queries."""
        print("\n" + "─" * 60)
        print("  🔍  Document–Query Similarity")
        print("─" * 60)

        documents = [
            "Employees are entitled to 15 days of paid annual leave per year.",
            "ChromaDB is an open-source vector database for AI applications.",
            "The company provides 16 weeks of paid parental leave.",
            "HNSW is the most popular indexing algorithm for vector search.",
        ]

        queries = [
            "How many vacation days do I get?",
            "What is a good vector database?",
        ]

        print("  Embedding documents…")
        doc_embeddings = self.embeddings_small.embed_documents(documents)

        for query in queries:
            print(f"\n  Query: \"{query}\"")
            query_emb = self.embeddings_small.embed_query(query)
            query_vec = np.array(query_emb)

            scores = []
            for i, doc in enumerate(documents):
                doc_vec = np.array(doc_embeddings[i])
                sim = cosine_similarity(query_vec, doc_vec)
                scores.append((sim, doc))

            scores.sort(reverse=True)
            for rank, (sim, doc) in enumerate(scores, 1):
                marker = "✅" if rank == 1 else "  "
                print(f"    {marker} [{sim:.4f}] {doc[:65]}…")

    # ── Semantic similarity demo ────────────────────────────────────
    def semantic_similarity_demo(self) -> None:
        """Show that semantically similar phrases have high similarity
        even when they use completely different words."""
        print("\n" + "─" * 60)
        print("  🧠  Semantic Similarity Demo")
        print("─" * 60)

        pairs = [
            # Semantically similar (different words)
            ("cancel subscription", "terminate policy"),
            ("I want to quit my job", "I'd like to resign from my position"),
            ("What is the refund policy?", "How do I get my money back?"),
            # Semantically different
            ("cancel subscription", "chocolate cake recipe"),
            ("employee vacation policy", "quantum physics equations"),
            # Lexically similar but semantically different
            ("bank of the river", "bank account balance"),
        ]

        print("  Comparing phrase pairs (text-embedding-3-small):\n")

        for text_a, text_b in pairs:
            emb_a = np.array(self.embeddings_small.embed_query(text_a))
            emb_b = np.array(self.embeddings_small.embed_query(text_b))
            sim = cosine_similarity(emb_a, emb_b)

            # Interpret the score
            if sim > 0.85:
                label = "🟢 Very similar"
            elif sim > 0.70:
                label = "🟡 Similar"
            elif sim > 0.50:
                label = "🟠 Somewhat related"
            else:
                label = "🔴 Different"

            print(f"    {label} [{sim:.4f}]")
            print(f"      A: \"{text_a}\"")
            print(f"      B: \"{text_b}\"")
            print()

        print("  💡  Embeddings capture meaning, not just word overlap!")
        print("      'cancel subscription' ≈ 'terminate policy' despite")
        print("      sharing zero words in common.")

    # ── Performance benchmark ───────────────────────────────────────
    def benchmark_performance(self) -> None:
        """Measure embedding speed and estimate costs."""
        print("\n" + "─" * 60)
        print("  ⚡  Embedding Performance Benchmark")
        print("─" * 60)

        # Generate test texts of varying sizes
        test_texts = [
            f"This is test document number {i}. It contains some sample text "
            f"about various topics including AI, machine learning, and data science. "
            f"The purpose is to benchmark embedding performance across different models."
            for i in range(20)
        ]

        # Rough token estimate (~4 chars per token)
        total_chars = sum(len(t) for t in test_texts)
        estimated_tokens = total_chars // 4

        for model_name, embedder in [
            ("text-embedding-3-small", self.embeddings_small),
            ("text-embedding-3-large", self.embeddings_large),
        ]:
            print(f"\n  Model: {model_name}")
            start = time.perf_counter()
            embeddings = embedder.embed_documents(test_texts)
            elapsed = time.perf_counter() - start

            tokens_per_sec = estimated_tokens / elapsed if elapsed > 0 else 0
            price = PRICING.get(model_name, 0)
            cost = (estimated_tokens / 1_000_000) * price

            print(f"    Documents       : {len(test_texts)}")
            print(f"    Dimensions      : {len(embeddings[0]):,}")
            print(f"    Time            : {elapsed:.3f}s")
            print(f"    Tokens (est.)   : {estimated_tokens:,}")
            print(f"    Tokens/sec      : {tokens_per_sec:,.0f}")
            print(f"    Cost estimate   : ${cost:.6f} "
                  f"(${price}/1M tokens)")

    # ── Vector math demo ────────────────────────────────────────────
    def vector_math_demo(self) -> None:
        """Demonstrate vector arithmetic on embeddings.

        Classic example: king - man + woman ≈ queen
        We use simpler, more reliable examples with our embedding model.
        """
        print("\n" + "─" * 60)
        print("  ➕  Vector Math Demo")
        print("─" * 60)

        words = {
            "king": "king",
            "queen": "queen",
            "man": "man",
            "woman": "woman",
            "paris": "Paris",
            "france": "France",
            "berlin": "Berlin",
            "germany": "Germany",
        }

        print("  Embedding words…")
        vecs: dict[str, np.ndarray] = {}
        for key, text in words.items():
            emb = self.embeddings_small.embed_query(text)
            vecs[key] = np.array(emb)

        # ── Addition / Subtraction ──────────────────────────────────
        print("\n  1️⃣  Vector Addition & Subtraction")
        print("      king - man + woman ≈ ?")

        result_vec = vecs["king"] - vecs["man"] + vecs["woman"]

        # Find closest word
        candidates = ["king", "queen", "man", "woman"]
        best_word, best_sim = "", -1.0
        for word in candidates:
            sim = cosine_similarity(result_vec, vecs[word])
            marker = ""
            if sim > best_sim:
                best_sim = sim
                best_word = word
            print(f"      → similarity to '{word}': {sim:.4f}")

        print(f"      ✅ Closest match: '{best_word}' (sim={best_sim:.4f})")

        # ── Country-capital analogy ─────────────────────────────────
        print("\n  2️⃣  Analogy: Paris:France :: Berlin:?")
        analogy_vec = vecs["paris"] - vecs["france"] + vecs["germany"]
        sim_berlin = cosine_similarity(analogy_vec, vecs["berlin"])
        sim_paris = cosine_similarity(analogy_vec, vecs["paris"])
        print(f"      → similarity to 'Berlin' : {sim_berlin:.4f}")
        print(f"      → similarity to 'Paris'  : {sim_paris:.4f}")

        # ── Magnitude and normalisation ─────────────────────────────
        print("\n  3️⃣  Vector Properties")
        vec = vecs["king"]
        print(f"      Dimensions  : {len(vec):,}")
        print(f"      Magnitude   : {np.linalg.norm(vec):.6f}")
        print(f"      Min value   : {vec.min():.6f}")
        print(f"      Max value   : {vec.max():.6f}")
        print(f"      Mean value  : {vec.mean():.6f}")
        print(f"      Std dev     : {vec.std():.6f}")

        # Normalised vectors have magnitude ≈ 1.0
        normalised = vec / np.linalg.norm(vec)
        print(f"      After normalisation, magnitude: {np.linalg.norm(normalised):.6f}")

        print("\n  💡  OpenAI embeddings are already normalised (magnitude ≈ 1.0),")
        print("      so cosine similarity = dot product for these vectors.")


# ════════════════════════════════════════════════════════════════════
# Main demonstration
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run all embedding explorations."""
    print("=" * 60)
    print("  Part 1.3 — Embeddings for RAG")
    print("=" * 60)

    try:
        explorer = EmbeddingExplorer()
    except EnvironmentError as e:
        print(f"\n❌  {e}")
        return

    # 1. Compare dimensions
    explorer.compare_dimensions()

    # 2. Document-query similarity
    explorer.document_query_similarity()

    # 3. Semantic similarity
    explorer.semantic_similarity_demo()

    # 4. Performance benchmark
    explorer.benchmark_performance()

    # 5. Vector math
    explorer.vector_math_demo()

    # Educational notes
    print("\n" + "═" * 60)
    print("💡  KEY TAKEAWAYS:")
    print("═" * 60)
    print("  1. Embeddings convert text → numerical vectors that capture meaning.")
    print("  2. text-embedding-3-small (1536-d) is cost-effective for most RAG apps.")
    print("  3. text-embedding-3-large (3072-d) captures finer distinctions.")
    print("  4. Cosine similarity measures semantic closeness (1.0 = identical).")
    print("  5. Embeddings handle synonyms: 'cancel' ≈ 'terminate'.")
    print("  6. Vector math reveals encoded relationships (king−man+woman ≈ queen).")
    print("  7. Always use the SAME embedding model for indexing and querying.\n")


if __name__ == "__main__":
    main()
