"""
Part 3, File 12: Reranking Pipeline — LLM-Based Cross-Encoder Reranking
========================================================================

Demonstrates a two-stage retrieval pipeline:
  Stage 1 — Retrieve k=10 *candidate* documents from ChromaDB (fast, cheap).
  Stage 2 — Rerank candidates with an LLM-based cross-encoder that scores
             each (query, document) pair on a 1-10 relevance scale.

Key concepts
------------
* Bi-encoder (embedding) retrieval is fast but approximate.
* Cross-encoder reranking is slower but far more accurate because it sees
  the query and document *together*.
* We simulate cross-encoder behaviour by asking the LLM to score relevance.
* Precision@3 measures what fraction of the top-3 results are relevant.

The pipeline prints a clear before/after comparison showing how reranking
reshuffles the initial results and improves precision.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ── Constants ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_reranking"
COLLECTION = "reranking_demo"

# Queries with ground-truth relevant snippets for evaluation
TEST_CASES: list[dict[str, Any]] = [
    {
        "query": "What is the parental leave policy?",
        "relevant_keywords": ["parental leave", "16 weeks", "primary caregiver", "secondary caregiver"],
    },
    {
        "query": "How does HNSW compare with other indexing algorithms?",
        "relevant_keywords": ["HNSW", "Hierarchical Navigable", "IVFFlat", "accuracy", "speed"],
    },
    {
        "query": "When must an employee provide a medical certificate for sick leave?",
        "relevant_keywords": ["3 consecutive days", "medical certificate", "sick leave"],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# RerankerPipeline
# ═══════════════════════════════════════════════════════════════════════
class RerankerPipeline:
    """Two-stage retrieval → reranking pipeline using an LLM cross-encoder."""

    RERANK_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a relevance scoring engine. Given a user query and a "
                "document excerpt, score how relevant the document is to the query "
                "on a scale of 1 to 10.\n\n"
                "Scoring guide:\n"
                "  1-3: Not relevant — the document does not address the query\n"
                "  4-6: Partially relevant — touches on the topic but lacks detail\n"
                "  7-9: Highly relevant — directly answers or closely relates\n"
                "  10:  Perfect match — the document fully and precisely answers the query\n\n"
                "Respond with ONLY a JSON object: {{\"score\": <int>, \"reason\": \"<brief reason>\"}}\n"
                "Do NOT include any other text.",
            ),
            (
                "human",
                "QUERY: {query}\n\n"
                "DOCUMENT:\n{document}\n\n"
                "Score this document's relevance (1-10):",
            ),
        ]
    )

    def __init__(self, initial_k: int = 10, final_k: int = 3) -> None:
        self.initial_k = initial_k
        self.final_k = final_k

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.vectorstore: Chroma | None = None
        self.documents: list[Document] = []

    # ── Build index ──────────────────────────────────────────────────
    def load_and_index(self, docs_dir: Path) -> None:
        """Load docs, chunk, and build ChromaDB index."""
        print("\n📂  Loading documents …")
        raw_docs: list[Document] = []
        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix in {".txt", ".md"}:
                loader = TextLoader(str(fpath), encoding="utf-8")
                raw_docs.extend(loader.load())
                print(f"   ✓ Loaded {fpath.name}")

        if not raw_docs:
            raise FileNotFoundError(f"No .txt/.md files in {docs_dir}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.documents = splitter.split_documents(raw_docs)
        print(f"   📄 Chunks created: {len(self.documents)}")

        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)

        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"   ✓ ChromaDB index ready ({len(self.documents)} vectors)\n")

    # ── Stage 1: Initial retrieval ───────────────────────────────────
    def retrieve_candidates(self, query: str) -> list[tuple[Document, float]]:
        """Return k=initial_k candidates from ChromaDB with similarity scores."""
        assert self.vectorstore is not None
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.initial_k
        )
        return results  # list of (Document, score)

    # ── Stage 2: LLM-based cross-encoder reranking ───────────────────
    def cross_encoder_rerank(
        self, query: str, docs: list[Document]
    ) -> list[dict[str, Any]]:
        """Score each doc against the query using the LLM and return sorted results.

        Returns a list of dicts: {doc, rerank_score, reason, original_rank}.
        """
        scored: list[dict[str, Any]] = []

        for rank, doc in enumerate(docs, 1):
            prompt = self.RERANK_PROMPT.format_messages(
                query=query,
                document=doc.page_content[:500],  # cap context length
            )
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            # Parse JSON score from LLM response
            try:
                parsed = json.loads(raw)
                score = int(parsed.get("score", 1))
                reason = parsed.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                # Fallback: try to extract a number
                match = re.search(r"\b(\d{1,2})\b", raw)
                score = int(match.group(1)) if match else 1
                reason = raw[:80]

            score = max(1, min(10, score))  # clamp 1-10
            scored.append(
                {
                    "doc": doc,
                    "rerank_score": score,
                    "reason": reason,
                    "original_rank": rank,
                }
            )

        # Sort descending by rerank score, break ties by original rank
        scored.sort(key=lambda x: (-x["rerank_score"], x["original_rank"]))
        return scored

    # ── Full pipeline ────────────────────────────────────────────────
    def run_pipeline(
        self, query: str, relevant_keywords: list[str]
    ) -> dict[str, Any]:
        """Execute the full retrieve → rerank pipeline and compute metrics."""
        print(f"  📋 Query: {query}")

        # Stage 1
        print(f"\n  Stage 1: Retrieving top-{self.initial_k} candidates …")
        t0 = time.perf_counter()
        candidates = self.retrieve_candidates(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        candidate_docs = [doc for doc, _score in candidates]
        candidate_scores = [score for _doc, score in candidates]

        print(f"  ⏱️  Retrieval: {retrieval_ms:.0f} ms")
        print(f"\n  {'Rank':<6} {'Sim Score':>9}  {'Relevant?':>9}  Content Preview")
        print(f"  {'─' * 68}")
        for i, (doc, sim) in enumerate(candidates, 1):
            is_rel = any(kw.lower() in doc.page_content.lower() for kw in relevant_keywords)
            marker = "  ✅" if is_rel else "  ❌"
            snippet = doc.page_content[:60].replace("\n", " ")
            print(f"  {i:<6} {sim:>9.4f}  {marker:<9}  {snippet}…")

        # Precision@3 before reranking
        top3_before = candidate_docs[: self.final_k]
        precision_before = sum(
            1
            for d in top3_before
            if any(kw.lower() in d.page_content.lower() for kw in relevant_keywords)
        ) / self.final_k

        # Stage 2
        print(f"\n  Stage 2: Reranking with LLM cross-encoder …")
        t0 = time.perf_counter()
        reranked = self.cross_encoder_rerank(query, candidate_docs)
        rerank_ms = (time.perf_counter() - t0) * 1000
        print(f"  ⏱️  Reranking: {rerank_ms:.0f} ms")

        print(f"\n  {'New Rank':<9} {'Score':>5} {'Was Rank':>8}  {'Relevant?':>9}  Reason")
        print(f"  {'─' * 72}")
        for new_rank, item in enumerate(reranked, 1):
            doc = item["doc"]
            is_rel = any(kw.lower() in doc.page_content.lower() for kw in relevant_keywords)
            marker = "  ✅" if is_rel else "  ❌"
            reason = item["reason"][:50] if item["reason"] else ""
            print(
                f"  {new_rank:<9} {item['rerank_score']:>5}/10"
                f" {item['original_rank']:>8}  {marker:<9}  {reason}"
            )

        # Precision@3 after reranking
        top3_after = [item["doc"] for item in reranked[: self.final_k]]
        precision_after = sum(
            1
            for d in top3_after
            if any(kw.lower() in d.page_content.lower() for kw in relevant_keywords)
        ) / self.final_k

        # Summary
        improvement = precision_after - precision_before
        print(f"\n  📊  Precision@{self.final_k}:")
        print(f"     Before reranking: {precision_before:.2f}  ({int(precision_before * self.final_k)}/{self.final_k} relevant)")
        print(f"     After  reranking: {precision_after:.2f}  ({int(precision_after * self.final_k)}/{self.final_k} relevant)")
        if improvement > 0:
            print(f"     🚀  Improvement: +{improvement:.2f}")
        elif improvement == 0:
            print(f"     →   No change (already optimal or same ranking)")
        else:
            print(f"     ⚠️   Decreased: {improvement:.2f}")

        return {
            "query": query,
            "precision_before": precision_before,
            "precision_after": precision_after,
            "improvement": improvement,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
        }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the reranking pipeline demonstration."""
    print("=" * 72)
    print("  Part 3 · File 12 — Reranking Pipeline (LLM Cross-Encoder)")
    print("=" * 72)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    pipeline = RerankerPipeline(initial_k=10, final_k=3)
    pipeline.load_and_index(DOCS_DIR)

    all_results: list[dict[str, Any]] = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'═' * 72}")
        print(f"  TEST CASE {i}")
        print(f"{'═' * 72}")
        result = pipeline.run_pipeline(tc["query"], tc["relevant_keywords"])
        all_results.append(result)

    # ── Overall summary ──────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  📊  OVERALL RERANKING RESULTS")
    print(f"{'═' * 72}")
    print(f"\n  {'Query':<55} {'P@3 Before':>10} {'P@3 After':>10} {'Δ':>6}")
    print(f"  {'─' * 85}")

    for r in all_results:
        q = r["query"][:52] + "…" if len(r["query"]) > 52 else r["query"]
        print(
            f"  {q:<55} {r['precision_before']:>10.2f} {r['precision_after']:>10.2f} "
            f"{r['improvement']:>+6.2f}"
        )

    avg_before = sum(r["precision_before"] for r in all_results) / len(all_results)
    avg_after = sum(r["precision_after"] for r in all_results) / len(all_results)
    avg_improvement = avg_after - avg_before
    avg_rerank_ms = sum(r["rerank_ms"] for r in all_results) / len(all_results)

    print(f"  {'─' * 85}")
    print(
        f"  {'AVERAGE':<55} {avg_before:>10.2f} {avg_after:>10.2f} {avg_improvement:>+6.2f}"
    )
    print(f"\n  ⏱️  Average reranking latency: {avg_rerank_ms:.0f} ms")
    print(
        "\n  💡  Reranking trades latency for precision. LLM-based reranking"
        "\n      is powerful but expensive — in production, consider a dedicated"
        "\n      cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM) for"
        "\n      lower latency and cost.\n"
    )

    # Cleanup
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹  Cleaned up temporary ChromaDB directory.\n")


if __name__ == "__main__":
    main()
