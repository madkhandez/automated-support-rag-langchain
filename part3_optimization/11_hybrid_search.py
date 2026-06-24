"""
Part 3, File 11: Hybrid Search — Dense + Sparse Retrieval
==========================================================

Demonstrates how to combine **dense** vector search (semantic similarity via
OpenAI embeddings + ChromaDB) with **sparse** BM25 keyword search (rank_bm25)
using LangChain's EnsembleRetriever.

Key concepts
------------
* Dense retrieval: finds semantically similar documents even when wording differs
* Sparse / BM25 retrieval: finds documents sharing exact keywords/terms
* Hybrid search: weighted combination for best of both worlds
* MRR (Mean Reciprocal Rank): measures where the first relevant result appears

Why hybrid?
-----------
Pure vector search can miss exact keyword matches ("PGVector" → cosine might
rank a generic embedding paragraph higher).  Pure BM25 misses paraphrases.
A 60/40 weighted ensemble typically beats either alone.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# BM25
from rank_bm25 import BM25Okapi

# ── Constants ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_hybrid_search"
COLLECTION = "hybrid_search_demo"

# Test queries and their expected relevant keywords (for MRR evaluation)
TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "How many days of annual leave do employees get?",
        "relevant_keywords": ["annual leave", "15 days", "vacation", "1.25 days per month"],
    },
    {
        "query": "What is the remote work policy at the company?",
        "relevant_keywords": ["remote work", "3 days per week", "core hours", "10 AM"],
    },
    {
        "query": "How does HNSW indexing compare to IVFFlat?",
        "relevant_keywords": ["HNSW", "IVFFlat", "speed", "accuracy", "recall"],
    },
    {
        "query": "What are the best vector databases for production?",
        "relevant_keywords": ["Pinecone", "Weaviate", "Qdrant", "Milvus", "ChromaDB"],
    },
    {
        "query": "What happens to unused vacation days at year end?",
        "relevant_keywords": ["carryover", "5 days", "forfeited", "December 31"],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# BM25 Retriever Wrapper
# ═══════════════════════════════════════════════════════════════════════
class BM25Retriever(BaseRetriever):
    """LangChain-compatible retriever backed by rank_bm25.BM25Okapi.

    We store the original Document objects and tokenise their page_content
    for BM25 scoring.  At retrieval time we score every document against
    the query tokens and return the top-k.
    """

    documents: list[Document] = []
    tokenized_docs: list[list[str]] = []
    bm25: Any = None  # BM25Okapi instance (not Pydantic-serialisable)
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(cls, documents: list[Document], k: int = 4) -> "BM25Retriever":
        """Build the BM25 index from a list of LangChain Documents."""
        tokenized = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        instance = cls(documents=documents, tokenized_docs=tokenized, bm25=bm25, k=k)
        return instance

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Return top-k documents by BM25 score."""
        if self.bm25 is None:
            return []
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        # Pair (score, index), sort descending, take top-k
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[Document] = []
        for idx, score in scored[: self.k]:
            doc = self.documents[idx].copy()
            doc.metadata["bm25_score"] = round(float(score), 4)
            results.append(doc)
        return results


# ═══════════════════════════════════════════════════════════════════════
# HybridSearchRetriever — orchestrates dense + sparse + ensemble
# ═══════════════════════════════════════════════════════════════════════
class HybridSearchRetriever:
    """Combines ChromaDB dense retrieval with BM25 sparse retrieval
    via LangChain's EnsembleRetriever."""

    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        k: int = 4,
    ) -> None:
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.k = k

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore: Chroma | None = None
        self.dense_retriever: Any = None
        self.sparse_retriever: BM25Retriever | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None
        self.documents: list[Document] = []

    # ── Loading & indexing ───────────────────────────────────────────
    def load_and_index(self, docs_dir: Path) -> None:
        """Load documents from *docs_dir*, chunk them, and build both
        dense and sparse indexes."""
        print("\n📂  Loading documents …")
        raw_docs: list[Document] = []

        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix in {".txt", ".md"}:
                loader = TextLoader(str(fpath), encoding="utf-8")
                raw_docs.extend(loader.load())
                print(f"   ✓ Loaded {fpath.name}")

        if not raw_docs:
            raise FileNotFoundError(f"No .txt/.md files found in {docs_dir}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.documents = splitter.split_documents(raw_docs)
        print(f"   📄 Total chunks: {len(self.documents)}")

        # ── Dense index (ChromaDB) ───────────────────────────────────
        print("\n🧠  Building dense (vector) index with ChromaDB …")
        # Clean up old data
        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)

        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        print(f"   ✓ Dense index ready  ({len(self.documents)} vectors)")

        # ── Sparse index (BM25) ─────────────────────────────────────
        print("\n🔤  Building sparse (BM25) index …")
        self.sparse_retriever = BM25Retriever.from_documents(self.documents, k=self.k)
        print(f"   ✓ BM25 index ready  ({len(self.documents)} documents)")

        # ── Ensemble ─────────────────────────────────────────────────
        print(
            f"\n⚖️   Creating EnsembleRetriever  "
            f"(dense={self.dense_weight}, sparse={self.sparse_weight})"
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[self.dense_weight, self.sparse_weight],
        )
        print("   ✓ Hybrid retriever ready")

    # ── Query helpers ────────────────────────────────────────────────
    def dense_search(self, query: str) -> list[Document]:
        """Return results from the dense (vector) retriever only."""
        assert self.dense_retriever is not None
        return self.dense_retriever.invoke(query)

    def sparse_search(self, query: str) -> list[Document]:
        """Return results from the sparse (BM25) retriever only."""
        assert self.sparse_retriever is not None
        return self.sparse_retriever.invoke(query)

    def hybrid_search(self, query: str) -> list[Document]:
        """Return results from the weighted ensemble retriever."""
        assert self.ensemble_retriever is not None
        return self.ensemble_retriever.invoke(query)

    # ── Evaluation ───────────────────────────────────────────────────
    @staticmethod
    def _doc_hash(doc: Document) -> str:
        return hashlib.md5(doc.page_content.encode()).hexdigest()[:10]

    @staticmethod
    def reciprocal_rank(results: list[Document], relevant_keywords: list[str]) -> float:
        """Compute Reciprocal Rank: 1/position of first relevant result.

        A result is considered relevant if it contains at least one of the
        *relevant_keywords* (case-insensitive).
        """
        for rank, doc in enumerate(results, start=1):
            text = doc.page_content.lower()
            if any(kw.lower() in text for kw in relevant_keywords):
                return 1.0 / rank
        return 0.0

    def run_comparison(self, test_queries: list[dict[str, Any]]) -> None:
        """Run all test queries through dense, sparse, and hybrid retrieval.
        Print detailed results and MRR scores."""

        dense_rrs: list[float] = []
        sparse_rrs: list[float] = []
        hybrid_rrs: list[float] = []

        for i, tq in enumerate(test_queries, 1):
            query = tq["query"]
            keywords = tq["relevant_keywords"]

            print(f"\n{'═' * 72}")
            print(f"  QUERY {i}: {query}")
            print(f"{'═' * 72}")

            # Dense
            t0 = time.perf_counter()
            dense_docs = self.dense_search(query)
            dense_ms = (time.perf_counter() - t0) * 1000
            dense_rr = self.reciprocal_rank(dense_docs, keywords)
            dense_rrs.append(dense_rr)

            print(f"\n  🧠 Dense (Semantic)  [{dense_ms:.1f} ms]  RR={dense_rr:.2f}")
            for j, doc in enumerate(dense_docs, 1):
                snippet = doc.page_content[:90].replace("\n", " ")
                src = Path(doc.metadata.get("source", "?")).name
                print(f"     {j}. [{src}]  {snippet}…")

            # Sparse
            t0 = time.perf_counter()
            sparse_docs = self.sparse_search(query)
            sparse_ms = (time.perf_counter() - t0) * 1000
            sparse_rr = self.reciprocal_rank(sparse_docs, keywords)
            sparse_rrs.append(sparse_rr)

            print(f"\n  🔤 Sparse (BM25)  [{sparse_ms:.1f} ms]  RR={sparse_rr:.2f}")
            for j, doc in enumerate(sparse_docs, 1):
                snippet = doc.page_content[:90].replace("\n", " ")
                bm25_score = doc.metadata.get("bm25_score", "n/a")
                src = Path(doc.metadata.get("source", "?")).name
                print(f"     {j}. [{src}]  bm25={bm25_score}  {snippet}…")

            # Hybrid
            t0 = time.perf_counter()
            hybrid_docs = self.hybrid_search(query)
            hybrid_ms = (time.perf_counter() - t0) * 1000
            hybrid_rr = self.reciprocal_rank(hybrid_docs, keywords)
            hybrid_rrs.append(hybrid_rr)

            print(f"\n  ⚖️  Hybrid (60/40)  [{hybrid_ms:.1f} ms]  RR={hybrid_rr:.2f}")
            for j, doc in enumerate(hybrid_docs, 1):
                snippet = doc.page_content[:90].replace("\n", " ")
                src = Path(doc.metadata.get("source", "?")).name
                print(f"     {j}. [{src}]  {snippet}…")

        # ── MRR summary ─────────────────────────────────────────────
        mrr_dense = sum(dense_rrs) / len(dense_rrs)
        mrr_sparse = sum(sparse_rrs) / len(sparse_rrs)
        mrr_hybrid = sum(hybrid_rrs) / len(hybrid_rrs)

        print(f"\n{'═' * 72}")
        print("  📊  MEAN RECIPROCAL RANK (MRR) SUMMARY")
        print(f"{'═' * 72}")
        print(f"  {'Method':<25} {'MRR':>8}  {'Interpretation'}")
        print(f"  {'─' * 60}")
        print(
            f"  {'🧠 Dense (Semantic)':<25} {mrr_dense:>8.3f}  "
            f"{'First relevant result at rank ~' + str(round(1/mrr_dense, 1)) if mrr_dense > 0 else 'No relevant results found'}"
        )
        print(
            f"  {'🔤 Sparse (BM25)':<25} {mrr_sparse:>8.3f}  "
            f"{'First relevant result at rank ~' + str(round(1/mrr_sparse, 1)) if mrr_sparse > 0 else 'No relevant results found'}"
        )
        print(
            f"  {'⚖️  Hybrid (60/40)':<25} {mrr_hybrid:>8.3f}  "
            f"{'First relevant result at rank ~' + str(round(1/mrr_hybrid, 1)) if mrr_hybrid > 0 else 'No relevant results found'}"
        )
        print()

        best = max(
            [("Dense", mrr_dense), ("Sparse", mrr_sparse), ("Hybrid", mrr_hybrid)],
            key=lambda x: x[1],
        )
        print(f"  🏆  Best performer: {best[0]} (MRR = {best[1]:.3f})")
        print(
            "\n  💡  MRR = 1.0 means the first result is always relevant."
            "\n      MRR = 0.5 means the relevant result is typically at rank 2."
            "\n      Higher is better — hybrid usually wins or ties.\n"
        )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the hybrid search demonstration end-to-end."""
    print("=" * 72)
    print("  Part 3 · File 11 — Hybrid Search (Dense + BM25)")
    print("=" * 72)

    # Load environment
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    # Build the retriever
    retriever = HybridSearchRetriever(dense_weight=0.6, sparse_weight=0.4, k=4)
    retriever.load_and_index(DOCS_DIR)

    # Run comparison benchmark
    retriever.run_comparison(TEST_QUERIES)

    # Cleanup
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹  Cleaned up temporary ChromaDB directory.\n")


if __name__ == "__main__":
    main()
