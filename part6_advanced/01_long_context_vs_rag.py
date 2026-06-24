"""
Part 6, File 1: Long Context vs. RAG — When to Use Each Approach

Demonstrates the trade-off between two strategies for grounding LLM answers:
  1. Long-Context (stuff ALL documents into the prompt)
  2. RAG (retrieve only the most relevant chunks)

Key insights:
  - Long context gives the model full visibility but costs more tokens and is slower.
  - RAG is cheaper and faster but may miss relevant context if retrieval fails.
  - The right choice depends on corpus size, cost budget, and latency requirements.

Decision framework:
  < 50 docs + need full context  → long context
  > 100 docs OR cost-sensitive    → RAG
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
CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "part6_01_long_ctx"

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: quick token estimate (4 chars ≈ 1 token for English text)
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Rough token estimate — good enough for cost comparisons."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# ═══════════════════════════════════════════════════════════════════════════
# LongContextVsRAG
# ═══════════════════════════════════════════════════════════════════════════

class LongContextVsRAG:
    """Benchmark harness comparing long-context stuffing vs. RAG retrieval."""

    # Pricing (per 1K tokens) — adjust if your model differs
    INPUT_COST_PER_1K = 0.0025    # gpt-4o input
    OUTPUT_COST_PER_1K = 0.01     # gpt-4o output

    def __init__(
        self,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 4,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer the question using ONLY the "
             "provided context. If the context doesn't contain the answer, say so.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])

        self.long_ctx_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. You have been given the COMPLETE set of "
             "documents below. Answer the question thoroughly using this information.\n\n"
             "Documents:\n{documents}"),
            ("human", "{question}"),
        ])

        self.chain_rag = self.rag_prompt | self.llm | StrOutputParser()
        self.chain_long = self.long_ctx_prompt | self.llm | StrOutputParser()

        # State
        self._vectorstore: Chroma | None = None
        self._all_docs: list[Document] = []
        self._chunks: list[Document] = []

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_documents(self, docs_dir: Path | str = DOCS_DIR) -> list[Document]:
        """Load .txt and .md files from *docs_dir*."""
        docs_dir = Path(docs_dir)
        all_docs: list[Document] = []

        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix in (".txt", ".md"):
                try:
                    loader = TextLoader(str(fpath), encoding="utf-8")
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source"] = fpath.name
                    all_docs.extend(loaded)
                    print(f"  📄 Loaded {fpath.name} ({len(loaded)} document(s))")
                except Exception as exc:
                    print(f"  ⚠️  Skipped {fpath.name}: {exc}")

        self._all_docs = all_docs
        print(f"  ✅ Total documents loaded: {len(all_docs)}")
        return all_docs

    # ------------------------------------------------------------------
    # RAG: index + retrieve + generate
    # ------------------------------------------------------------------

    def _build_vectorstore(self) -> Chroma:
        """Split documents into chunks and build a ChromaDB vectorstore."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._chunks = splitter.split_documents(self._all_docs)
        print(f"  🔪 Split into {len(self._chunks)} chunks "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        # Use a unique collection name to avoid collisions
        collection_name = f"part6_01_{uuid.uuid4().hex[:8]}"

        vectorstore = Chroma.from_documents(
            documents=self._chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"  🗄️  ChromaDB collection '{collection_name}' created "
              f"with {len(self._chunks)} vectors")
        self._vectorstore = vectorstore
        return vectorstore

    def rag_approach(self, question: str, docs: list[Document] | None = None) -> dict[str, Any]:
        """
        RAG approach: embed chunks → retrieve top-k → generate answer.

        Returns dict with answer, latency, token counts, and cost.
        """
        if docs:
            self._all_docs = docs

        if self._vectorstore is None:
            self._build_vectorstore()

        assert self._vectorstore is not None

        t0 = time.perf_counter()

        # Retrieve
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        retrieved = retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in retrieved)

        # Generate
        answer = self.chain_rag.invoke({"context": context, "question": question})
        elapsed = time.perf_counter() - t0

        # Token accounting
        input_tokens = _estimate_tokens(context) + _estimate_tokens(question) + 60  # system overhead
        output_tokens = _estimate_tokens(answer)
        cost = (input_tokens / 1000) * self.INPUT_COST_PER_1K + \
               (output_tokens / 1000) * self.OUTPUT_COST_PER_1K

        return {
            "approach": "RAG",
            "answer": answer,
            "latency_s": round(elapsed, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "chunks_retrieved": len(retrieved),
            "sources": [d.metadata.get("source", "?") for d in retrieved],
        }

    # ------------------------------------------------------------------
    # Long-context: stuff ALL docs into prompt
    # ------------------------------------------------------------------

    def long_context_approach(self, question: str, docs: list[Document] | None = None) -> dict[str, Any]:
        """
        Long-context approach: concatenate ALL document text and feed to the LLM.

        Returns dict with answer, latency, token counts, and cost.
        """
        if docs:
            self._all_docs = docs

        all_text = "\n\n===== DOCUMENT SEPARATOR =====\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in self._all_docs
        )

        t0 = time.perf_counter()
        answer = self.chain_long.invoke({"documents": all_text, "question": question})
        elapsed = time.perf_counter() - t0

        input_tokens = _estimate_tokens(all_text) + _estimate_tokens(question) + 60
        output_tokens = _estimate_tokens(answer)
        cost = (input_tokens / 1000) * self.INPUT_COST_PER_1K + \
               (output_tokens / 1000) * self.OUTPUT_COST_PER_1K

        return {
            "approach": "Long Context",
            "answer": answer,
            "latency_s": round(elapsed, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "docs_included": len(self._all_docs),
            "sources": [d.metadata.get("source", "?") for d in self._all_docs],
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, questions: list[str]) -> list[dict[str, Any]]:
        """
        Run both approaches on every question and return a list of paired results.
        """
        results: list[dict[str, Any]] = []
        for i, q in enumerate(questions, 1):
            print(f"\n{'─' * 70}")
            print(f"  Question {i}: {q}")
            print(f"{'─' * 70}")

            rag = self.rag_approach(q)
            long_ctx = self.long_context_approach(q)

            results.append({"question": q, "rag": rag, "long_context": long_ctx})

            # Pretty-print side-by-side
            self._print_comparison(rag, long_ctx)

        return results

    # ------------------------------------------------------------------
    # Decision framework
    # ------------------------------------------------------------------

    @staticmethod
    def decide(num_docs: int, need_full_context: bool, cost_sensitive: bool) -> str:
        """
        Decision framework: recommend long-context or RAG.

        Rules:
          - < 50 docs AND need full context          → Long Context
          - > 100 docs OR cost-sensitive              → RAG
          - 50-100 docs, cost-tolerant, partial ctx   → Hybrid or Long Context
        """
        if num_docs > 100 or cost_sensitive:
            return "RAG"
        if num_docs < 50 and need_full_context:
            return "Long Context"
        # Middle ground
        if need_full_context:
            return "Long Context (with caution — approaching token limits)"
        return "RAG (sufficient for partial-context needs)"

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    @staticmethod
    def _print_comparison(rag: dict[str, Any], long_ctx: dict[str, Any]) -> None:
        """Print a side-by-side benchmark table."""
        header = f"{'Metric':<25} {'RAG':>20} {'Long Context':>20}"
        sep = "─" * 67
        print(f"\n  {sep}")
        print(f"  {header}")
        print(f"  {sep}")

        rows = [
            ("Latency (s)", f"{rag['latency_s']:.3f}", f"{long_ctx['latency_s']:.3f}"),
            ("Input tokens", f"{rag['input_tokens']:,}", f"{long_ctx['input_tokens']:,}"),
            ("Output tokens", f"{rag['output_tokens']:,}", f"{long_ctx['output_tokens']:,}"),
            ("Total tokens", f"{rag['total_tokens']:,}", f"{long_ctx['total_tokens']:,}"),
            ("Est. cost ($)", f"${rag['estimated_cost_usd']:.6f}", f"${long_ctx['estimated_cost_usd']:.6f}"),
        ]
        for label, v_rag, v_long in rows:
            print(f"  {label:<25} {v_rag:>20} {v_long:>20}")

        print(f"  {sep}")

        # Winner labels
        latency_winner = "RAG ✓" if rag["latency_s"] < long_ctx["latency_s"] else "Long Ctx ✓"
        cost_winner = "RAG ✓" if rag["estimated_cost_usd"] < long_ctx["estimated_cost_usd"] else "Long Ctx ✓"
        print(f"  {'Faster:':<25} {latency_winner:>20}")
        print(f"  {'Cheaper:':<25} {cost_winner:>20}")
        print()

        # Truncated answers
        for result in (rag, long_ctx):
            snippet = result["answer"][:200].replace("\n", " ")
            if len(result["answer"]) > 200:
                snippet += "…"
            print(f"  📝 {result['approach']} answer: {snippet}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the Long-Context vs. RAG benchmark end-to-end."""
    print("=" * 70)
    print("  Part 6.1 — Long Context vs. RAG Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Validate environment
    # ------------------------------------------------------------------
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not found in environment.")
        print("   Copy .env.example → .env and add your key.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Instantiate benchmark harness
    # ------------------------------------------------------------------
    print("\n📦 Initialising LongContextVsRAG harness …")
    harness = LongContextVsRAG(
        model_name=os.getenv("LLM_MODEL", "gpt-4o"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=4,
    )

    # ------------------------------------------------------------------
    # 3. Load sample documents
    # ------------------------------------------------------------------
    print(f"\n📂 Loading documents from {DOCS_DIR} …")
    harness.load_documents(DOCS_DIR)

    # ------------------------------------------------------------------
    # 4. Benchmark questions
    # ------------------------------------------------------------------
    questions = [
        "How many days of annual leave do full-time employees get?",
        "What vector database indexing algorithm offers the best speed-accuracy trade-off?",
        "What is the remote work policy and how many days can employees work from home?",
    ]

    print("\n🏁 Running benchmark …")
    results = harness.benchmark(questions)

    # ------------------------------------------------------------------
    # 5. Aggregate summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📊 Aggregate Summary")
    print("=" * 70)

    total_rag_cost = sum(r["rag"]["estimated_cost_usd"] for r in results)
    total_lc_cost = sum(r["long_context"]["estimated_cost_usd"] for r in results)
    avg_rag_latency = sum(r["rag"]["latency_s"] for r in results) / len(results)
    avg_lc_latency = sum(r["long_context"]["latency_s"] for r in results) / len(results)

    print(f"  Questions asked:              {len(results)}")
    print(f"  Avg RAG latency:              {avg_rag_latency:.3f}s")
    print(f"  Avg Long-Context latency:     {avg_lc_latency:.3f}s")
    print(f"  Total RAG cost:               ${total_rag_cost:.6f}")
    print(f"  Total Long-Context cost:      ${total_lc_cost:.6f}")

    savings_pct = (1 - total_rag_cost / total_lc_cost) * 100 if total_lc_cost > 0 else 0
    print(f"  RAG cost savings:             {savings_pct:.1f}%")

    # ------------------------------------------------------------------
    # 6. Decision framework demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🧭 Decision Framework")
    print("=" * 70)

    scenarios = [
        (10, True, False, "10 docs, need full context, cost-tolerant"),
        (200, False, True, "200 docs, partial context, cost-sensitive"),
        (75, True, False, "75 docs, need full context, cost-tolerant"),
        (30, False, True, "30 docs, partial context, cost-sensitive"),
    ]
    for n, full, cost, desc in scenarios:
        rec = LongContextVsRAG.decide(n, full, cost)
        print(f"  • {desc}")
        print(f"    → Recommendation: {rec}")

    # ------------------------------------------------------------------
    # 7. Clean up
    # ------------------------------------------------------------------
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        print(f"\n🧹 Cleaned up ChromaDB directory: {CHROMA_DIR}")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
