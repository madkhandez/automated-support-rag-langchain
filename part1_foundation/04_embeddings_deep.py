"""
Part 1 — Embeddings Deep Dive
==============================
Deep exploration of embedding models: dimension impact, vocabulary gap,
batch embedding with caching, cost calculation, and model comparison.
"""

import os
import sys
import time
import hashlib
import shelve
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class EmbeddingDeepDive:
    """Deep dive into embedding model behaviour, caching, and cost analysis."""

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        self.small_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.large_model = OpenAIEmbeddings(model="text-embedding-3-large")
        # Pricing per 1M tokens (as of 2026)
        self.pricing = {
            "text-embedding-3-small": 0.02,   # $0.02 / 1M tokens
            "text-embedding-3-large": 0.13,   # $0.13 / 1M tokens
        }

    # ── Dimension impact ────────────────────────────────────────────
    def dimension_impact_demo(self):
        """Show how embedding dimensions affect retrieval quality."""
        print("\n" + "=" * 70)
        print("DIMENSION IMPACT ON RETRIEVAL QUALITY")
        print("=" * 70)

        query = "How do I set up a vector database?"
        documents = [
            "ChromaDB is an open-source vector database for AI applications.",
            "PostgreSQL with PGVector extension supports vector similarity search.",
            "The weather forecast calls for rain tomorrow afternoon.",
            "Vector databases store high-dimensional embeddings for fast retrieval.",
        ]

        print(f"\nQuery: '{query}'")
        print(f"\nDocuments:")
        for i, doc in enumerate(documents, 1):
            print(f"  [{i}] {doc}")

        # Embed with small model (1536 dims)
        print(f"\n{'Model':<30} {'Dimensions':<12} {'Top Match':<8} {'Scores'}")
        print("-" * 80)

        for name, model, dims in [
            ("text-embedding-3-small", self.small_model, 1536),
            ("text-embedding-3-large", self.large_model, 3072),
        ]:
            q_emb = model.embed_query(query)
            d_embs = model.embed_documents(documents)

            scores = []
            for d_emb in d_embs:
                q_arr = np.array(q_emb)
                d_arr = np.array(d_emb)
                sim = np.dot(q_arr, d_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(d_arr))
                scores.append(float(sim))

            top_idx = int(np.argmax(scores))
            score_str = ", ".join(f"{s:.4f}" for s in scores)
            print(f"  {name:<28} {dims:<12} Doc [{top_idx + 1}]   [{score_str}]")

        print("\n💡 Higher dimensions capture finer semantic nuances, but cost more.")

    # ── Vocabulary gap ──────────────────────────────────────────────
    def vocabulary_gap_demo(self):
        """Demonstrate the vocabulary gap problem."""
        print("\n" + "=" * 70)
        print("VOCABULARY GAP PROBLEM")
        print("=" * 70)

        pairs = [
            ("cancel my subscription", "termination of service policy"),
            ("cancel my subscription", "discontinue your account"),
            ("cancel my subscription", "cancel my subscription"),
            ("how to get a refund", "reimbursement process and procedures"),
            ("delete my data", "data erasure and removal policy"),
        ]

        print(f"\n{'Query':<30} {'Document':<45} {'Similarity'}")
        print("-" * 85)

        for query, doc in pairs:
            q_emb = np.array(self.small_model.embed_query(query))
            d_emb = np.array(self.small_model.embed_query(doc))
            sim = float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)))
            bar = "█" * int(sim * 30)
            print(f"  {query:<28} {doc:<43} {sim:.4f} {bar}")

        print("\n💡 Embedding models bridge the vocabulary gap — synonyms score high similarity")
        print("   even when exact keywords don't match. This is why embeddings beat keyword search.")

    # ── Batch embedding with progress ───────────────────────────────
    def batch_embed(self, texts: list[str], model_name: str = "small",
                    batch_size: int = 100) -> list[list[float]]:
        """Embed texts in batches with progress tracking and rate limiting."""
        model = self.small_model if model_name == "small" else self.large_model
        all_embeddings = []
        total = len(texts)
        start = time.time()

        print(f"\n📦 Batch embedding {total} texts (batch_size={batch_size})...")

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = model.embed_documents(batch)
            all_embeddings.extend(batch_embs)

            progress = min(i + batch_size, total)
            pct = progress / total * 100
            elapsed = time.time() - start
            rate = progress / elapsed if elapsed > 0 else 0
            print(f"  ✅ [{progress}/{total}] {pct:.0f}% — {rate:.1f} texts/sec")

            # Simple rate limiting: 50ms pause between batches
            if i + batch_size < total:
                time.sleep(0.05)

        elapsed = time.time() - start
        print(f"\n  ⏱️  Total time: {elapsed:.2f}s ({total / elapsed:.1f} texts/sec)")
        return all_embeddings

    # ── Embedding cache ─────────────────────────────────────────────
    def cached_embed_demo(self):
        """Demonstrate disk-based embedding caching with shelve."""
        print("\n" + "=" * 70)
        print("EMBEDDING CACHING (DISK-BASED)")
        print("=" * 70)

        cache_path = os.path.join(os.path.dirname(__file__), ".embedding_cache")
        texts = [
            "What is retrieval augmented generation?",
            "How do vector databases work?",
            "Explain the chunking process for RAG",
            "What is retrieval augmented generation?",  # duplicate → cache hit
            "How do vector databases work?",            # duplicate → cache hit
        ]

        hits = 0
        misses = 0
        embeddings = []

        with shelve.open(cache_path) as cache:
            for text in texts:
                key = hashlib.sha256(text.encode()).hexdigest()
                if key in cache:
                    embeddings.append(cache[key])
                    hits += 1
                    print(f"  ✅ CACHE HIT:  '{text[:50]}...'")
                else:
                    emb = self.small_model.embed_query(text)
                    cache[key] = emb
                    embeddings.append(emb)
                    misses += 1
                    print(f"  ❌ CACHE MISS: '{text[:50]}...'")

        total = hits + misses
        print(f"\n  📊 Cache Stats: {hits} hits, {misses} misses, "
              f"hit rate = {hits / total * 100:.0f}%")
        print(f"  💾 Cache stored at: {cache_path}")

        # Cleanup cache files
        for ext in ["", ".db", ".dir", ".bak", ".dat"]:
            path = cache_path + ext
            if os.path.exists(path):
                os.remove(path)

    # ── Cost calculation ────────────────────────────────────────────
    def cost_analysis(self):
        """Calculate real embedding costs for a document collection."""
        print("\n" + "=" * 70)
        print("EMBEDDING COST ANALYSIS")
        print("=" * 70)

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Simulate a document collection
        sample_texts = [
            "LangChain is an open-source framework for building LLM applications." * 10,
            "Vector databases provide fast similarity search over embeddings." * 10,
            "RAG combines retrieval with generation for better answers." * 10,
            "ChromaDB stores vectors locally for development purposes." * 10,
            "Production systems often use PGVector with Supabase." * 10,
        ]

        total_tokens = sum(len(enc.encode(t)) for t in sample_texts)

        print(f"\n  📄 Documents: {len(sample_texts)}")
        print(f"  🔤 Total tokens: {total_tokens:,}")
        print(f"\n  {'Model':<30} {'Price/1M tokens':<18} {'Cost':<12} {'Dimensions'}")
        print("  " + "-" * 75)

        for model_name, price in self.pricing.items():
            cost = (total_tokens / 1_000_000) * price
            dims = 1536 if "small" in model_name else 3072
            print(f"  {model_name:<30} ${price:<16.2f} ${cost:<10.6f} {dims}")

        # Projections
        print(f"\n  📈 Cost Projections (1000 documents × 10 chunks × 200 tokens each):")
        proj_tokens = 1000 * 10 * 200
        for model_name, price in self.pricing.items():
            proj_cost = (proj_tokens / 1_000_000) * price
            print(f"     {model_name}: {proj_tokens:,} tokens → ${proj_cost:.4f}")

    # ── Model comparison ────────────────────────────────────────────
    def model_comparison(self):
        """Compare text-embedding-3-small vs text-embedding-3-large."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: small vs large")
        print("=" * 70)

        test_texts = [
            "Machine learning algorithms process data to find patterns.",
            "Deep learning uses neural networks with many layers.",
            "What are the best practices for training ML models?",
        ]

        results = {}
        for name, model in [("small", self.small_model), ("large", self.large_model)]:
            start = time.time()
            embs = model.embed_documents(test_texts)
            elapsed = time.time() - start
            dims = len(embs[0])
            results[name] = {
                "latency": elapsed,
                "dimensions": dims,
                "texts_per_sec": len(test_texts) / elapsed if elapsed > 0 else 0,
            }

        print(f"\n  {'Metric':<25} {'Small (3-small)':<20} {'Large (3-large)':<20}")
        print("  " + "-" * 65)
        print(f"  {'Dimensions':<25} {results['small']['dimensions']:<20} "
              f"{results['large']['dimensions']:<20}")
        print(f"  {'Latency (sec)':<25} {results['small']['latency']:<20.3f} "
              f"{results['large']['latency']:<20.3f}")
        print(f"  {'Texts/sec':<25} {results['small']['texts_per_sec']:<20.1f} "
              f"{results['large']['texts_per_sec']:<20.1f}")
        print(f"  {'Price/1M tokens':<25} {'$0.02':<20} {'$0.13':<20}")
        print(f"  {'Cost ratio':<25} {'1x':<20} {'6.5x':<20}")

        print("\n  💡 Recommendation:")
        print("     • Use text-embedding-3-small for most use cases (best cost/quality)")
        print("     • Use text-embedding-3-large only when retrieval precision is critical")


def main():
    """Run all embedding deep dive demonstrations."""
    print("🔬 EMBEDDINGS DEEP DIVE")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in .env to run this module.")
        print("   cp .env.example .env && edit .env")
        return

    explorer = EmbeddingDeepDive()

    explorer.vocabulary_gap_demo()
    explorer.dimension_impact_demo()
    explorer.cached_embed_demo()
    explorer.cost_analysis()
    explorer.model_comparison()

    # Batch embed demo with sample texts
    sample = [f"Document chunk number {i} about AI and machine learning." for i in range(10)]
    explorer.batch_embed(sample, model_name="small", batch_size=5)

    print("\n" + "=" * 70)
    print("✅ Embeddings Deep Dive complete!")


if __name__ == "__main__":
    main()
