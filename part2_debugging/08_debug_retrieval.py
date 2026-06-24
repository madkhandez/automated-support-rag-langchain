"""
08_debug_retrieval.py — Retrieval Debugging Toolkit

Provides the RetrievalDebugger class with 5 diagnostic tools for
understanding and fixing retrieval quality issues in RAG pipelines.

Tools:
  1. analyze_query_embedding()   — inspect embedding vector statistics
  2. compare_retrieval_strategies() — test different k values for quality/quantity tradeoff
  3. debug_metadata_filters()    — test with/without metadata filters
  4. score_distribution()        — visualise similarity score histogram
  5. find_retrieval_gaps()       — identify queries that consistently retrieve poorly

Usage:
  python 08_debug_retrieval.py

Requires:
  - OPENAI_API_KEY in .env
  - docs/ folder with sample documents at ../docs/
"""

import os
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Load environment ─────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / ".chroma_debug_retrieval"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RetrievalDebugger
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RetrievalDebugger:
    """Diagnostic tools for debugging retrieval quality in RAG pipelines.

    Builds a local ChromaDB vector store from sample documents and
    provides methods to probe, measure, and compare retrieval behaviour.
    """

    def __init__(self, rebuild: bool = False) -> None:
        """Initialize the debugger with embeddings and a local vector store.

        Args:
            rebuild: If True, delete and rebuild the vector store from scratch.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment. "
                "Copy .env.example → .env and add your key."
            )

        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Build / load vector store
        if rebuild and CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)
            print("  🗑️  Cleared existing vector store.")

        self.vectorstore = self._build_vectorstore()
        print(f"  ✅ Vector store ready — {self.vectorstore._collection.count()} documents indexed.")

    # ── Vector store construction ─────────────────────────────
    def _build_vectorstore(self) -> Chroma:
        """Build a ChromaDB vector store from the docs/ folder."""
        collection_name = "debug_retrieval"

        # Try loading existing store first
        store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        if store._collection.count() > 0:
            return store

        # Load and chunk documents
        print("  📄 Loading documents from docs/ folder...")
        documents: list[Document] = []

        for filepath in sorted(DOCS_DIR.glob("*")):
            if filepath.suffix in (".txt", ".md"):
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs = loader.load()
                # Add rich metadata for filter testing
                for doc in docs:
                    doc.metadata["source"] = filepath.name
                    doc.metadata["file_type"] = filepath.suffix.lstrip(".")
                    if "policy" in filepath.name.lower():
                        doc.metadata["category"] = "policy"
                        doc.metadata["department"] = "HR"
                    elif "tech" in filepath.name.lower():
                        doc.metadata["category"] = "technical"
                        doc.metadata["department"] = "Engineering"
                    else:
                        doc.metadata["category"] = "general"
                        doc.metadata["department"] = "General"
                documents.extend(docs)

        if not documents:
            print("  ⚠️  No documents found in docs/. Creating synthetic documents.")
            documents = self._create_synthetic_docs()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"  📦 Split into {len(chunks)} chunks.")

        # Store in ChromaDB
        store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
        return store

    @staticmethod
    def _create_synthetic_docs() -> list[Document]:
        """Create synthetic documents when docs/ folder is empty."""
        return [
            Document(
                page_content="Full-time employees are entitled to 15 days of paid annual leave per calendar year. Annual leave accrues at a rate of 1.25 days per month.",
                metadata={"source": "policy.txt", "category": "policy", "department": "HR", "file_type": "txt"},
            ),
            Document(
                page_content="Vector databases are specialized data storage systems designed to efficiently index, store, and retrieve high-dimensional vector embeddings.",
                metadata={"source": "tech.md", "category": "technical", "department": "Engineering", "file_type": "md"},
            ),
            Document(
                page_content="ChromaDB is an open-source, lightweight vector database ideal for development and prototyping. It runs locally without a separate server process.",
                metadata={"source": "tech.md", "category": "technical", "department": "Engineering", "file_type": "md"},
            ),
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 1: Analyze Query Embedding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def analyze_query_embedding(self, query: str) -> dict[str, Any]:
        """Inspect the embedding vector for a query — dimensions, magnitude, and statistics.

        This helps you understand what the embedding model "sees" and whether
        the query is likely to retrieve well.

        Args:
            query: The search query to embed and analyze.

        Returns:
            Dict of embedding statistics.
        """
        print(f"\n  ── Analyzing Embedding for: \"{query}\" ────────────")

        embedding = self.embeddings.embed_query(query)
        vec = np.array(embedding)

        stats = {
            "query": query,
            "dimensions": len(embedding),
            "magnitude": float(np.linalg.norm(vec)),
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            "median": float(np.median(vec)),
            "nonzero_pct": float(np.count_nonzero(vec) / len(vec) * 100),
            "top_5_indices": list(np.argsort(np.abs(vec))[-5:][::-1]),
        }

        print(f"    Dimensions:       {stats['dimensions']}")
        print(f"    Magnitude (L2):   {stats['magnitude']:.6f}")
        print(f"    Mean:             {stats['mean']:.6f}")
        print(f"    Std Dev:          {stats['std']:.6f}")
        print(f"    Min / Max:        {stats['min']:.6f} / {stats['max']:.6f}")
        print(f"    Median:           {stats['median']:.6f}")
        print(f"    Non-zero dims:    {stats['nonzero_pct']:.1f}%")
        print(f"    Top-5 active dims (by |value|): {stats['top_5_indices']}")

        # Show first 10 values as a preview
        preview = [f"{v:.4f}" for v in embedding[:10]]
        print(f"    First 10 values:  [{', '.join(preview)}, ...]")

        # Sanity check: is the vector normalized?
        if abs(stats["magnitude"] - 1.0) < 0.05:
            print("    ✅ Vector is approximately unit-normalized (expected for OpenAI).")
        else:
            print(f"    ⚠️  Vector magnitude is {stats['magnitude']:.4f}, not ~1.0.")

        return stats

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 2: Compare Retrieval Strategies
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def compare_retrieval_strategies(
        self,
        query: str,
        k_values: Optional[list[int]] = None,
    ) -> dict[int, list[dict]]:
        """Run similarity search with different k values to see quality vs quantity tradeoff.

        Args:
            query: The search query.
            k_values: List of k values to test. Defaults to [1, 3, 5, 10].

        Returns:
            Dict mapping k → list of result dicts (content preview, score, metadata).
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        print(f"\n  ── Comparing Retrieval Strategies ─────────────────")
        print(f"    Query: \"{query}\"")
        print(f"    Testing k values: {k_values}")

        results: dict[int, list[dict]] = {}

        for k in k_values:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k
                )
            except Exception:
                # Fallback if relevance scores aren't available
                docs = self.vectorstore.similarity_search(query, k=k)
                docs_with_scores = [(doc, 0.0) for doc in docs]

            print(f"\n    ── k={k} {'─' * 50}")

            run_results = []
            scores_list = []
            for i, (doc, score) in enumerate(docs_with_scores):
                preview = doc.page_content.replace("\n", " ")[:60]
                source = doc.metadata.get("source", "unknown")
                result_dict = {
                    "rank": i + 1,
                    "score": score,
                    "source": source,
                    "preview": preview,
                }
                run_results.append(result_dict)
                scores_list.append(score)
                print(f"      [{i+1}] score={score:.4f}  src={source:<20s}  \"{preview}...\"")

            if scores_list:
                avg_score = statistics.mean(scores_list)
                min_score = min(scores_list)
                quality_drop = scores_list[0] - scores_list[-1] if len(scores_list) > 1 else 0
                print(f"      ── Avg={avg_score:.4f}  Min={min_score:.4f}  "
                      f"Top-Bottom gap={quality_drop:.4f}")

                if quality_drop > 0.15:
                    print(f"      ⚠️  Large quality gap ({quality_drop:.4f}) — "
                          f"last results are significantly less relevant.")

            results[k] = run_results

        # Overall recommendation
        print(f"\n    ── Recommendation ──")
        print("    Start with k=3-5 for most use cases.")
        print("    Use k=1-2 for highly specific factual queries.")
        print("    Use k=7-10 only for broad exploratory queries with post-filtering.")

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 3: Debug Metadata Filters
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def debug_metadata_filters(
        self,
        query: str,
        k: int = 5,
    ) -> dict[str, list[dict]]:
        """Test retrieval with and without metadata filters to see their impact.

        Runs the query three ways:
          1. No filter (baseline)
          2. Filter by category="policy"
          3. Filter by category="technical"

        Args:
            query: The search query.
            k: Number of results per run.

        Returns:
            Dict mapping filter_name → list of result dicts.
        """
        print(f"\n  ── Debugging Metadata Filters ─────────────────────")
        print(f"    Query: \"{query}\"")
        print(f"    k={k}")

        filter_configs: list[tuple[str, Optional[dict]]] = [
            ("No Filter (baseline)", None),
            ("category='policy'", {"category": "policy"}),
            ("category='technical'", {"category": "technical"}),
        ]

        all_results: dict[str, list[dict]] = {}

        for filter_name, filter_dict in filter_configs:
            print(f"\n    ── {filter_name} {'─' * max(1, 45 - len(filter_name))}")

            try:
                if filter_dict:
                    docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                        query, k=k, filter=filter_dict
                    )
                else:
                    docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                        query, k=k
                    )
            except Exception as e:
                print(f"      ❌ Error: {e}")
                all_results[filter_name] = []
                continue

            results = []
            for i, (doc, score) in enumerate(docs_with_scores):
                preview = doc.page_content.replace("\n", " ")[:55]
                meta_str = ", ".join(f"{k}={v}" for k, v in doc.metadata.items()
                                     if k in ("category", "department", "source"))
                result_dict = {
                    "rank": i + 1,
                    "score": score,
                    "metadata": dict(doc.metadata),
                    "preview": preview,
                }
                results.append(result_dict)
                print(f"      [{i+1}] score={score:.4f}  [{meta_str}]")
                print(f"          \"{preview}...\"")

            if not results:
                print("      (no results — filter may be too restrictive)")

            all_results[filter_name] = results

        # Analysis
        print(f"\n    ── Analysis ──")
        baseline_count = len(all_results.get("No Filter (baseline)", []))
        for filter_name, results in all_results.items():
            if filter_name == "No Filter (baseline)":
                continue
            filtered_count = len(results)
            if filtered_count == 0 and baseline_count > 0:
                print(f"    ⚠️  '{filter_name}' returned 0 results but baseline returned "
                      f"{baseline_count}. Check if metadata values are correct.")
            elif filtered_count > 0:
                avg_score = statistics.mean(r["score"] for r in results)
                baseline_avg = statistics.mean(r["score"] for r in all_results["No Filter (baseline)"]) if all_results.get("No Filter (baseline)") else 0
                if avg_score > baseline_avg:
                    print(f"    ✅ '{filter_name}' improved avg score: "
                          f"{baseline_avg:.4f} → {avg_score:.4f}")
                else:
                    print(f"    ℹ️  '{filter_name}' avg score: {avg_score:.4f} "
                          f"(baseline: {baseline_avg:.4f})")

        return all_results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 4: Score Distribution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def score_distribution(self, query: str, k: int = 20) -> dict[str, Any]:
        """Show a histogram-like distribution of similarity scores.

        Helps identify whether there's a clear relevance cliff (good)
        or a gradual decline (bad — hard to set a threshold).

        Args:
            query: The search query.
            k: Number of results to retrieve (higher = more distribution data).

        Returns:
            Dict with score statistics and bucket counts.
        """
        print(f"\n  ── Score Distribution ─────────────────────────────")
        print(f"    Query: \"{query}\"")
        print(f"    k={k}")

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k
            )
        except Exception:
            docs = self.vectorstore.similarity_search(query, k=k)
            docs_with_scores = [(doc, 0.0) for doc in docs]

        scores = [score for _, score in docs_with_scores]

        if not scores:
            print("    ❌ No results retrieved.")
            return {"scores": [], "buckets": {}}

        # Score statistics
        stats = {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

        print(f"\n    Statistics:")
        print(f"      Count:   {stats['count']}")
        print(f"      Mean:    {stats['mean']:.4f}")
        print(f"      Median:  {stats['median']:.4f}")
        print(f"      Std Dev: {stats['stdev']:.4f}")
        print(f"      Range:   [{stats['min']:.4f}, {stats['max']:.4f}]")

        # Build histogram buckets
        buckets = {
            "0.90-1.00 (Excellent)": 0,
            "0.80-0.90 (Good)":      0,
            "0.70-0.80 (Fair)":      0,
            "0.60-0.70 (Marginal)":  0,
            "0.50-0.60 (Poor)":      0,
            "0.00-0.50 (Noise)":     0,
        }

        for score in scores:
            if score >= 0.90:
                buckets["0.90-1.00 (Excellent)"] += 1
            elif score >= 0.80:
                buckets["0.80-0.90 (Good)"] += 1
            elif score >= 0.70:
                buckets["0.70-0.80 (Fair)"] += 1
            elif score >= 0.60:
                buckets["0.60-0.70 (Marginal)"] += 1
            elif score >= 0.50:
                buckets["0.50-0.60 (Poor)"] += 1
            else:
                buckets["0.00-0.50 (Noise)"] += 1

        print(f"\n    Histogram:")
        max_bar_width = 40
        max_count = max(buckets.values()) if max(buckets.values()) > 0 else 1
        for label, count in buckets.items():
            bar_len = int(count / max_count * max_bar_width)
            bar = "█" * bar_len
            print(f"      {label:<25s}  {bar} ({count})")

        # Score curve (show each score as a dot plot)
        print(f"\n    Score Curve (each dot = one result):")
        for i, score in enumerate(scores):
            dot_pos = int(score * 50)
            line = " " * dot_pos + "●"
            print(f"      [{i+1:>2}] {score:.4f} |{line}")

        # Detect cliff
        if len(scores) > 2:
            diffs = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
            max_diff = max(diffs)
            cliff_idx = diffs.index(max_diff) + 1
            if max_diff > 0.05:
                print(f"\n    📉 Relevance cliff detected after result #{cliff_idx} "
                      f"(score drop of {max_diff:.4f})")
                print(f"    → Consider setting k={cliff_idx} or threshold ≥ "
                      f"{scores[cliff_idx]:.3f}")
            else:
                print(f"\n    ℹ️  Gradual score decline (max drop: {max_diff:.4f}). "
                      f"No clear cliff — use a similarity threshold instead of fixed k.")

        return {"scores": scores, "stats": stats, "buckets": buckets}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 5: Find Retrieval Gaps
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def find_retrieval_gaps(
        self,
        queries: list[str],
        k: int = 3,
        threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        """Find questions that consistently get low-quality retrievals.

        These are "gaps" in your knowledge base — questions your users ask
        but your documents don't cover well.

        Args:
            queries: List of test queries to evaluate.
            k: Number of results to retrieve per query.
            threshold: Minimum acceptable average score.

        Returns:
            List of gap dicts sorted by severity (worst gaps first).
        """
        print(f"\n  ── Finding Retrieval Gaps ─────────────────────────")
        print(f"    Testing {len(queries)} queries, k={k}, threshold={threshold}")

        gaps: list[dict[str, Any]] = []

        for query in queries:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k
                )
            except Exception:
                docs = self.vectorstore.similarity_search(query, k=k)
                docs_with_scores = [(doc, 0.0) for doc in docs]

            scores = [score for _, score in docs_with_scores]
            avg_score = statistics.mean(scores) if scores else 0.0
            top_score = max(scores) if scores else 0.0

            gap_info = {
                "query": query,
                "avg_score": avg_score,
                "top_score": top_score,
                "num_results": len(scores),
                "scores": scores,
                "is_gap": avg_score < threshold,
            }
            gaps.append(gap_info)

            status = "❌ GAP" if gap_info["is_gap"] else "✅ OK"
            print(f"    {status}  avg={avg_score:.4f}  top={top_score:.4f}  \"{query[:50]}\"")

        # Sort by severity (lowest avg score first)
        gaps.sort(key=lambda g: g["avg_score"])

        # Summary
        gap_count = sum(1 for g in gaps if g["is_gap"])
        print(f"\n    Summary: {gap_count}/{len(queries)} queries have retrieval gaps "
              f"(avg score < {threshold})")

        if gap_count > 0:
            print(f"\n    Worst gaps (need new documents or better chunking):")
            for g in gaps[:5]:
                if g["is_gap"]:
                    print(f"      ❌ avg={g['avg_score']:.4f}  \"{g['query'][:55]}\"")

            print(f"\n    Recommendations:")
            print(f"      1. Add documents that answer the gap queries")
            print(f"      2. Try query rewriting to bridge vocabulary gaps")
            print(f"      3. Consider smaller chunk sizes for better granularity")
            print(f"      4. Add metadata to enable filtered retrieval")

        return gaps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Run all retrieval debugging tools with sample queries."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Retrieval Debugger — Diagnostic Toolkit                 ║")
    print("║        5 tools for understanding retrieval quality             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        debugger = RetrievalDebugger(rebuild=True)
    except EnvironmentError as e:
        print(f"\n❌ Setup Error: {e}")
        sys.exit(1)

    # ── Tool 1: Analyze Query Embedding ──────────────────────
    print("\n\n🔧 TOOL 1: Analyze Query Embedding")
    print("=" * 60)
    queries_to_analyze = [
        "What is the annual leave policy?",
        "How do vector databases work?",
    ]
    for q in queries_to_analyze:
        try:
            debugger.analyze_query_embedding(q)
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # ── Tool 2: Compare Retrieval Strategies ─────────────────
    print("\n\n🔧 TOOL 2: Compare Retrieval Strategies")
    print("=" * 60)
    try:
        debugger.compare_retrieval_strategies(
            query="How many vacation days do employees get?",
            k_values=[1, 3, 5, 10],
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 3: Debug Metadata Filters ───────────────────────
    print("\n\n🔧 TOOL 3: Debug Metadata Filters")
    print("=" * 60)
    try:
        debugger.debug_metadata_filters(
            query="What is ChromaDB?",
            k=5,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 4: Score Distribution ───────────────────────────
    print("\n\n🔧 TOOL 4: Score Distribution")
    print("=" * 60)
    try:
        debugger.score_distribution(
            query="employee sick leave policy",
            k=15,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 5: Find Retrieval Gaps ──────────────────────────
    print("\n\n🔧 TOOL 5: Find Retrieval Gaps")
    print("=" * 60)
    test_queries = [
        "What is the annual leave policy?",
        "How many sick days do I get?",
        "What are the best vector databases?",
        "How does HNSW indexing work?",
        "What is ACME's policy on cryptocurrency payments?",   # Gap — not in docs
        "How do I set up Kubernetes for RAG?",                 # Gap — not in docs
        "What is the bereavement leave policy?",
        "How to configure PGVector in production?",
    ]
    try:
        debugger.find_retrieval_gaps(
            queries=test_queries,
            k=3,
            threshold=0.70,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    print(f"\n{'═' * 60}")
    print("✅ All retrieval debugging tools complete.")
    print("   Review output above to identify retrieval quality issues.")


if __name__ == "__main__":
    main()
