"""
Part 3, File 13: Multi-Query Retrieval — Broader Recall via Query Expansion
=============================================================================

Demonstrates LangChain's MultiQueryRetriever which:
  1. Takes one user question
  2. Uses an LLM to generate N alternative phrasings / perspectives
  3. Runs retrieval for *each* phrasing
  4. Deduplicates results by content hash
  5. Returns the merged, unique document set

Why?
----
A single query embedding may miss relevant docs phrased differently.  By
searching with multiple phrasings we cast a wider net, often increasing
recall from ~4 unique docs to 8-12 unique docs.

This file also implements a manual multi-query workflow alongside the
built-in MultiQueryRetriever so you can see exactly what happens at each step.
"""

from __future__ import annotations

import hashlib
import logging
import os
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
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# ── Constants ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_multi_query"
COLLECTION = "multi_query_demo"
NUM_ALTERNATIVE_QUERIES = 4

TEST_QUERIES = [
    "What is the company's remote work policy?",
    "How do vector databases store and retrieve data?",
    "What leave benefits does the company offer for new parents?",
]


# ═══════════════════════════════════════════════════════════════════════
# MultiQueryRAG
# ═══════════════════════════════════════════════════════════════════════
class MultiQueryRAG:
    """Multi-query retrieval with deduplication and comparison metrics."""

    QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant helping to improve document retrieval. "
                "Given the user's question, generate {num_queries} alternative "
                "versions of the question that approach the topic from different "
                "angles or use different vocabulary.\n\n"
                "Rules:\n"
                "- Each alternative should be meaningfully different\n"
                "- Cover synonyms, related concepts, and different specificity levels\n"
                "- Output ONLY the alternative questions, one per line\n"
                "- Do NOT number them or add bullet points\n"
                "- Do NOT include the original question",
            ),
            ("human", "Original question: {question}"),
        ]
    )

    def __init__(self, num_alternatives: int = NUM_ALTERNATIVE_QUERIES) -> None:
        self.num_alternatives = num_alternatives
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.vectorstore: Chroma | None = None
        self.multi_retriever: MultiQueryRetriever | None = None

    # ── Build index ──────────────────────────────────────────────────
    def load_and_index(self, docs_dir: Path) -> None:
        """Load, chunk, and index documents into ChromaDB."""
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
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"   📄 Chunks: {len(chunks)}")

        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"   ✓ ChromaDB index ready ({len(chunks)} vectors)\n")

        # Build LangChain's built-in MultiQueryRetriever
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

    # ── Query generation ─────────────────────────────────────────────
    def generate_alternative_queries(self, question: str) -> list[str]:
        """Use the LLM to generate alternative phrasings of *question*."""
        chain = self.QUERY_GEN_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke(
            {"question": question, "num_queries": self.num_alternatives}
        )
        alternatives = [
            line.strip()
            for line in result.strip().split("\n")
            if line.strip()
        ]
        return alternatives[: self.num_alternatives]

    # ── Single query retrieval ───────────────────────────────────────
    def single_query_retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Standard single-query retrieval from ChromaDB."""
        assert self.vectorstore is not None
        return self.vectorstore.similarity_search(query, k=k)

    # ── Manual multi-query retrieval ─────────────────────────────────
    @staticmethod
    def _content_hash(doc: Document) -> str:
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def manual_multi_query_retrieve(
        self, question: str
    ) -> dict[str, Any]:
        """Step-by-step multi-query retrieval with full visibility.

        Returns a dict with all intermediate data for display.
        """
        # Step 1: Generate alternatives
        print("  Step 1: Generating alternative queries …")
        t0 = time.perf_counter()
        alternatives = self.generate_alternative_queries(question)
        gen_ms = (time.perf_counter() - t0) * 1000

        all_queries = [question] + alternatives
        print(f"  ⏱️  Query generation: {gen_ms:.0f} ms")
        print(f"\n  📝  Queries ({len(all_queries)} total):")
        for i, q in enumerate(all_queries):
            label = "ORIGINAL" if i == 0 else f"ALT {i}"
            print(f"     [{label:>8}]  {q}")

        # Step 2: Retrieve for each query
        print(f"\n  Step 2: Retrieving documents for each query (k=4 each) …")
        all_docs: list[Document] = []
        per_query_results: list[list[Document]] = []
        t0 = time.perf_counter()

        for i, q in enumerate(all_queries):
            docs = self.single_query_retrieve(q, k=4)
            per_query_results.append(docs)
            all_docs.extend(docs)
            label = "ORIGINAL" if i == 0 else f"ALT {i}"
            print(f"     [{label:>8}]  → {len(docs)} docs retrieved")

        retrieval_ms = (time.perf_counter() - t0) * 1000
        print(f"  ⏱️  Total retrieval: {retrieval_ms:.0f} ms")

        # Step 3: Deduplicate
        print(f"\n  Step 3: Deduplicating results …")
        seen_hashes: set[str] = set()
        unique_docs: list[Document] = []
        duplicates = 0

        for doc in all_docs:
            h = self._content_hash(doc)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_docs.append(doc)
            else:
                duplicates += 1

        print(f"     Total retrieved:  {len(all_docs)}")
        print(f"     Duplicates found: {duplicates}")
        print(f"     Unique documents: {len(unique_docs)}")

        return {
            "original_query": question,
            "alternatives": alternatives,
            "all_queries": all_queries,
            "per_query_results": per_query_results,
            "all_docs_count": len(all_docs),
            "unique_docs": unique_docs,
            "duplicates": duplicates,
            "gen_ms": gen_ms,
            "retrieval_ms": retrieval_ms,
        }

    # ── LangChain built-in multi-query retrieval ─────────────────────
    def langchain_multi_query_retrieve(self, question: str) -> list[Document]:
        """Use LangChain's built-in MultiQueryRetriever."""
        assert self.multi_retriever is not None
        return self.multi_retriever.invoke(question)

    # ── Full comparison ──────────────────────────────────────────────
    def run_comparison(self, question: str) -> dict[str, Any]:
        """Compare single-query vs multi-query retrieval side by side."""
        print(f"\n  📋  Query: \"{question}\"")
        print(f"  {'─' * 65}")

        # Single query
        print("\n  ── SINGLE QUERY RETRIEVAL ──")
        t0 = time.perf_counter()
        single_docs = self.single_query_retrieve(question, k=4)
        single_ms = (time.perf_counter() - t0) * 1000
        print(f"  Results: {len(single_docs)} documents  ({single_ms:.0f} ms)")
        for i, doc in enumerate(single_docs, 1):
            src = Path(doc.metadata.get("source", "?")).name
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"     {i}. [{src}]  {snippet}…")

        # Manual multi-query
        print("\n  ── MULTI-QUERY RETRIEVAL (Manual) ──")
        multi_result = self.manual_multi_query_retrieve(question)

        # Show unique docs
        print(f"\n  📑  Final unique document set ({len(multi_result['unique_docs'])} docs):")
        for i, doc in enumerate(multi_result["unique_docs"], 1):
            src = Path(doc.metadata.get("source", "?")).name
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"     {i}. [{src}]  {snippet}…")

        # Summary
        single_count = len(single_docs)
        multi_count = len(multi_result["unique_docs"])
        gain = multi_count - single_count
        gain_pct = (gain / single_count * 100) if single_count > 0 else 0

        print(f"\n  📊  Comparison:")
        print(f"     Single-query docs:  {single_count}")
        print(f"     Multi-query docs:   {multi_count}")
        print(f"     Recall gain:        +{gain} docs (+{gain_pct:.0f}%)")
        print(f"     Queries generated:  {len(multi_result['all_queries'])}")
        print(f"     Duplicates removed: {multi_result['duplicates']}")

        return {
            "query": question,
            "single_count": single_count,
            "multi_count": multi_count,
            "gain": gain,
            "gain_pct": gain_pct,
            "alternatives": multi_result["alternatives"],
        }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the multi-query retrieval demonstration."""
    print("=" * 72)
    print("  Part 3 · File 13 — Multi-Query Retrieval")
    print("=" * 72)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    # Enable logging to see LangChain's internal query generation
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    rag = MultiQueryRAG(num_alternatives=NUM_ALTERNATIVE_QUERIES)
    rag.load_and_index(DOCS_DIR)

    all_results: list[dict[str, Any]] = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'═' * 72}")
        print(f"  TEST CASE {i}")
        print(f"{'═' * 72}")
        result = rag.run_comparison(query)
        all_results.append(result)

    # ── Overall summary ──────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  📊  OVERALL MULTI-QUERY RETRIEVAL SUMMARY")
    print(f"{'═' * 72}")
    print(f"\n  {'Query':<45} {'Single':>7} {'Multi':>7} {'Gain':>7}")
    print(f"  {'─' * 70}")

    for r in all_results:
        q = r["query"][:42] + "…" if len(r["query"]) > 42 else r["query"]
        print(
            f"  {q:<45} {r['single_count']:>7} "
            f"{r['multi_count']:>7} {'+' + str(r['gain']):>7}"
        )

    avg_single = sum(r["single_count"] for r in all_results) / len(all_results)
    avg_multi = sum(r["multi_count"] for r in all_results) / len(all_results)
    avg_gain = avg_multi - avg_single

    print(f"  {'─' * 70}")
    print(f"  {'AVERAGE':<45} {avg_single:>7.1f} {avg_multi:>7.1f} {'+' + f'{avg_gain:.1f}':>7}")
    print(
        f"\n  💡  Multi-query retrieval increased recall by an average of"
        f" {avg_gain:.1f} unique documents."
        f"\n      This broader context helps the LLM produce more comprehensive answers."
        f"\n      Trade-off: {NUM_ALTERNATIVE_QUERIES + 1}x retrieval calls + 1 LLM call"
        f" for query generation.\n"
    )

    # Cleanup
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹  Cleaned up temporary ChromaDB directory.\n")


if __name__ == "__main__":
    main()
