"""
Part 6, File 2: Contextual Retrieval — Anthropic's Technique for Better Chunks

Problem:
  When documents are split into chunks, each chunk loses the *document-level context*
  it came from. For example, a chunk that says "The policy allows 16 weeks" doesn't
  mention *which* policy or *which company*. This makes retrieval less precise because
  the embedding doesn't capture the full meaning.

Solution (Contextual Retrieval):
  Before embedding a chunk, use an LLM to generate a 2-3 sentence *context summary*
  of where the chunk fits within the full document, then prepend that summary to the
  chunk text. The enriched chunk embeds more meaningfully.

This file benchmarks standard chunking vs. contextual chunking on retrieval precision.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain v1 imports
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "part6_02_contextual"

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


# ═══════════════════════════════════════════════════════════════════════════
# ContextualRetrieval
# ═══════════════════════════════════════════════════════════════════════════

class ContextualRetrieval:
    """
    Implements Anthropic's Contextual Retrieval technique.

    For each chunk:
      1. Send the *full document* + the *chunk* to an LLM.
      2. Ask the LLM to write a short context summary for the chunk.
      3. Prepend the summary to the chunk text.
      4. Embed the enriched chunk.

    This dramatically improves retrieval precision because the embedding now
    captures both the local content and the document-level context.
    """

    CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a document analyst. Given the FULL DOCUMENT and a specific CHUNK "
         "extracted from it, write a concise 2-3 sentence context summary that:\n"
         "  1. Identifies the document's title/topic\n"
         "  2. Explains where this chunk fits within the document\n"
         "  3. Provides any broader context needed to understand the chunk\n\n"
         "Write ONLY the context summary — no preamble, no labels."),
        ("human",
         "FULL DOCUMENT:\n{full_doc}\n\n---\n\nCHUNK:\n{chunk}"),
    ])

    def __init__(
        self,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 4,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.context_chain = self.CONTEXT_PROMPT | self.llm | StrOutputParser()

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Core: prepend context
    # ------------------------------------------------------------------

    def prepend_context(self, chunk: str, full_doc: str) -> str:
        """
        Use an LLM to generate a 2-3 sentence context summary and prepend
        it to *chunk* before embedding.

        Returns:
            The enriched chunk: "<context summary>\\n\\n<original chunk>"
        """
        context_summary = self.context_chain.invoke({
            "full_doc": full_doc,
            "chunk": chunk,
        })
        enriched = f"{context_summary.strip()}\n\n{chunk}"
        return enriched

    # ------------------------------------------------------------------
    # Pipeline: standard chunking
    # ------------------------------------------------------------------

    def standard_chunking(self, docs: list[Document]) -> list[Document]:
        """Split documents into chunks WITHOUT any context enrichment."""
        chunks = self._splitter.split_documents(docs)
        for c in chunks:
            c.metadata["chunking"] = "standard"
        return chunks

    # ------------------------------------------------------------------
    # Pipeline: contextual chunking
    # ------------------------------------------------------------------

    def contextual_chunking(self, docs: list[Document]) -> list[Document]:
        """
        Split documents into chunks, then enrich each with a prepended
        context summary from the LLM.
        """
        chunks = self._splitter.split_documents(docs)
        enriched_chunks: list[Document] = []

        for i, chunk in enumerate(chunks):
            # Find the full doc this chunk came from
            source = chunk.metadata.get("source", "")
            full_doc_text = ""
            for doc in docs:
                if doc.metadata.get("source", "") == source:
                    full_doc_text = doc.page_content
                    break

            if not full_doc_text:
                full_doc_text = chunk.page_content  # fallback

            print(f"    Enriching chunk {i + 1}/{len(chunks)} "
                  f"(source: {source}) …")

            enriched_text = self.prepend_context(chunk.page_content, full_doc_text)

            enriched_doc = Document(
                page_content=enriched_text,
                metadata={
                    **chunk.metadata,
                    "chunking": "contextual",
                    "original_chunk": chunk.page_content[:100] + "…",
                },
            )
            enriched_chunks.append(enriched_doc)

        return enriched_chunks

    # ------------------------------------------------------------------
    # Build vectorstore
    # ------------------------------------------------------------------

    def _build_store(self, chunks: list[Document], label: str) -> Chroma:
        """Build a ChromaDB vectorstore from a list of document chunks."""
        collection_name = f"part6_02_{label}_{uuid.uuid4().hex[:6]}"
        store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"    🗄️  Collection '{collection_name}': {len(chunks)} vectors")
        return store

    # ------------------------------------------------------------------
    # Evaluate retrieval precision
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        store: Chroma,
        queries_with_expected: list[dict[str, Any]],
    ) -> float:
        """
        Evaluate retrieval precision.

        Each entry in *queries_with_expected* is:
            {"query": str, "expected_keywords": list[str]}

        A retrieved chunk is considered *relevant* if it contains at least
        one of the expected keywords (case-insensitive).

        Returns the average precision across all queries.
        """
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        total_precision = 0.0
        for entry in queries_with_expected:
            query = entry["query"]
            keywords = [kw.lower() for kw in entry["expected_keywords"]]
            results = retriever.invoke(query)

            relevant = 0
            for doc in results:
                text_lower = doc.page_content.lower()
                if any(kw in text_lower for kw in keywords):
                    relevant += 1

            precision = relevant / len(results) if results else 0.0
            total_precision += precision

        avg_precision = total_precision / len(queries_with_expected)
        return avg_precision

    # ------------------------------------------------------------------
    # Full comparison pipeline
    # ------------------------------------------------------------------

    def compare(
        self,
        docs: list[Document],
        queries_with_expected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run both standard and contextual chunking, build vectorstores,
        and compare retrieval precision.
        """
        results: dict[str, Any] = {}

        # — Standard —
        print("\n  📋 Standard Chunking")
        t0 = time.perf_counter()
        std_chunks = self.standard_chunking(docs)
        std_store = self._build_store(std_chunks, "standard")
        std_precision = self.evaluate_retrieval(std_store, queries_with_expected)
        std_time = time.perf_counter() - t0

        results["standard"] = {
            "num_chunks": len(std_chunks),
            "precision": round(std_precision, 4),
            "time_s": round(std_time, 2),
        }

        # — Contextual —
        print("\n  🧠 Contextual Chunking (LLM-enriched)")
        t0 = time.perf_counter()
        ctx_chunks = self.contextual_chunking(docs)
        ctx_store = self._build_store(ctx_chunks, "contextual")
        ctx_precision = self.evaluate_retrieval(ctx_store, queries_with_expected)
        ctx_time = time.perf_counter() - t0

        results["contextual"] = {
            "num_chunks": len(ctx_chunks),
            "precision": round(ctx_precision, 4),
            "time_s": round(ctx_time, 2),
        }

        return results


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Demonstrate contextual retrieval and compare with standard chunking."""
    print("=" * 70)
    print("  Part 6.2 — Contextual Retrieval (Anthropic Technique)")
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
    all_docs: list[Document] = []
    for fpath in sorted(DOCS_DIR.iterdir()):
        if fpath.suffix in (".txt", ".md"):
            try:
                loader = TextLoader(str(fpath), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = fpath.name
                all_docs.extend(loaded)
                print(f"  📄 Loaded {fpath.name}")
            except Exception as exc:
                print(f"  ⚠️  Skipped {fpath.name}: {exc}")

    print(f"  ✅ Total documents: {len(all_docs)}")

    # ------------------------------------------------------------------
    # 3. Define evaluation queries with expected keywords
    # ------------------------------------------------------------------
    queries_with_expected = [
        {
            "query": "How many days of paid annual leave do employees receive?",
            "expected_keywords": ["annual leave", "15 days", "vacation", "1.25 days"],
        },
        {
            "query": "What is the parental leave policy for primary caregivers?",
            "expected_keywords": ["parental leave", "16 weeks", "primary caregiver", "adoption"],
        },
        {
            "query": "Which vector database indexing algorithm is best for production?",
            "expected_keywords": ["hnsw", "hierarchical", "indexing", "speed", "recall"],
        },
        {
            "query": "What are the remote work rules at ACME?",
            "expected_keywords": ["remote", "3 days", "core hours", "10 am"],
        },
        {
            "query": "How does ChromaDB compare to PGVector?",
            "expected_keywords": ["chromadb", "pgvector", "postgresql", "lightweight"],
        },
    ]

    # ------------------------------------------------------------------
    # 4. Instantiate and run comparison
    # ------------------------------------------------------------------
    cr = ContextualRetrieval(
        model_name=os.getenv("LLM_MODEL", "gpt-4o"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=400,
        chunk_overlap=80,
        top_k=4,
    )

    print("\n🏁 Running comparison …")
    results = cr.compare(all_docs, queries_with_expected)

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📊 Retrieval Precision Comparison")
    print("=" * 70)

    header = f"{'Metric':<30} {'Standard':>15} {'Contextual':>15}"
    sep = "─" * 62
    print(f"\n  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    std = results["standard"]
    ctx = results["contextual"]

    rows = [
        ("Chunks", str(std["num_chunks"]), str(ctx["num_chunks"])),
        ("Avg Precision", f"{std['precision']:.2%}", f"{ctx['precision']:.2%}"),
        ("Processing time (s)", f"{std['time_s']:.1f}", f"{ctx['time_s']:.1f}"),
    ]
    for label, v_std, v_ctx in rows:
        print(f"  {label:<30} {v_std:>15} {v_ctx:>15}")

    print(f"  {sep}")

    improvement = ctx["precision"] - std["precision"]
    print(f"\n  Precision improvement: {improvement:+.2%}")

    if improvement > 0:
        print("  ✅ Contextual chunking improved retrieval precision!")
    elif improvement == 0:
        print("  ➡️  No change — try with larger/more diverse document sets.")
    else:
        print("  ⚠️  Standard was better — contextual may not help for very short docs.")

    # ------------------------------------------------------------------
    # 6. Show example enriched chunk
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🔍 Example: Standard Chunk vs. Contextual Chunk")
    print("=" * 70)

    # Grab a representative chunk for display
    std_chunks = cr.standard_chunking(all_docs)
    if std_chunks:
        sample = std_chunks[0]
        print(f"\n  📄 Standard chunk (source: {sample.metadata.get('source', '?')}):")
        print(f"  {'-' * 60}")
        for line in sample.page_content[:300].split("\n"):
            print(f"    {line}")
        print(f"    …" if len(sample.page_content) > 300 else "")

        print(f"\n  🧠 Contextual chunk (after LLM enrichment):")
        print(f"  {'-' * 60}")
        full_doc = next(
            (d.page_content for d in all_docs
             if d.metadata.get("source") == sample.metadata.get("source")),
            sample.page_content,
        )
        enriched = cr.prepend_context(sample.page_content, full_doc)
        for line in enriched[:500].split("\n"):
            print(f"    {line}")
        if len(enriched) > 500:
            print(f"    …")

    # ------------------------------------------------------------------
    # 7. Clean up
    # ------------------------------------------------------------------
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        print(f"\n🧹 Cleaned up ChromaDB directory: {CHROMA_DIR}")

    print("\n✅ Contextual retrieval comparison complete!")


if __name__ == "__main__":
    main()
