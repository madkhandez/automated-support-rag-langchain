"""
Part 3, File 15: Optimized RAG Pipeline — All Optimizations Combined
=====================================================================

Combines every optimization from Part 3 into a single, benchmarkable pipeline:

  1. **Hybrid Search** — dense (semantic) + sparse (BM25) via EnsembleRetriever
  2. **Multi-Query Retrieval** — LLM-generated alternative queries for broader recall
  3. **Reranking** — LLM-based cross-encoder scoring (1-10) to re-sort candidates
  4. **Token Budgeting** — tiktoken-based dynamic context window with map-reduce fallback

The file runs a side-by-side benchmark comparing a **Basic RAG** pipeline (single
query → top-4 ChromaDB results → answer) against the **Optimized RAG** pipeline
and prints a comparison table with metrics:

  • Retrieval precision (fraction of top-k results that are relevant)
  • Answer quality (LLM self-evaluation on a 1-10 scale)
  • Latency (end-to-end wall time)
  • Token usage (context tokens consumed)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import EnsembleRetriever

# BM25
from rank_bm25 import BM25Okapi

# ── Constants ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_optimized_rag"
COLLECTION = "optimized_rag_demo"

# Benchmark test cases — query + ground-truth keywords for precision measurement
BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "query": "What is the company's parental leave policy?",
        "relevant_keywords": [
            "parental leave", "16 weeks", "primary caregiver",
            "secondary caregiver", "6 weeks", "adoption",
        ],
    },
    {
        "query": "How does HNSW indexing work and what are its trade-offs?",
        "relevant_keywords": [
            "HNSW", "Hierarchical Navigable", "speed", "accuracy",
            "IVFFlat", "recall",
        ],
    },
    {
        "query": "What are the rules for remote work and core hours?",
        "relevant_keywords": [
            "remote work", "3 days per week", "core hours",
            "10 AM", "3 PM", "manager approval",
        ],
    },
    {
        "query": "What happens when an employee resigns or is terminated?",
        "relevant_keywords": [
            "termination", "2 weeks notice", "resignation",
            "unused accrued", "daily rate",
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Lightweight BM25 Retriever (reused from file 11)
# ═══════════════════════════════════════════════════════════════════════
class BM25SparseRetriever(BaseRetriever):
    """BM25-backed retriever compatible with LangChain's retriever interface."""

    documents: list[Document] = []
    tokenized_docs: list[list[str]] = []
    bm25: Any = None
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 4) -> "BM25SparseRetriever":
        tokenized = [d.page_content.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        return cls(documents=docs, tokenized_docs=tokenized, bm25=bm25, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        if self.bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[Document] = []
        for idx, score in ranked[: self.k]:
            doc = self.documents[idx].copy()
            doc.metadata["bm25_score"] = round(float(score), 4)
            results.append(doc)
        return results


# ═══════════════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════════════
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the context does not contain the answer, say so.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate {n} alternative phrasings of the user's question. "
            "Use different vocabulary, angles, and specificity levels. "
            "Output one question per line, no numbering or bullets.",
        ),
        ("human", "{question}"),
    ]
)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Score how relevant this document is to the query (1-10). "
            "Respond ONLY with JSON: {{\"score\": <int>, \"reason\": \"<brief>\"}}",
        ),
        (
            "human",
            "Query: {query}\n\nDocument:\n{document}\n\nScore:",
        ),
    ]
)

QUALITY_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an answer quality evaluator. Given a question and an answer, "
            "score the answer from 1 to 10 for completeness, accuracy, and clarity. "
            "Respond ONLY with JSON: {{\"score\": <int>, \"reason\": \"<brief>\"}}",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer: {answer}\n\nScore:",
        ),
    ]
)


# ═══════════════════════════════════════════════════════════════════════
# OptimizedRAGPipeline
# ═══════════════════════════════════════════════════════════════════════
class OptimizedRAGPipeline:
    """Production-grade RAG pipeline combining all Part 3 optimizations."""

    def __init__(
        self,
        llm_max_tokens: int = 8000,
        safety_margin: float = 0.8,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        num_alt_queries: int = 3,
        initial_k: int = 10,
        final_k: int = 4,
    ) -> None:
        # Models
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        # Config
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.num_alt_queries = num_alt_queries
        self.initial_k = initial_k
        self.final_k = final_k

        # Token budgeting
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.llm_max_tokens = llm_max_tokens
        self.safety_margin = safety_margin
        self.total_budget = int(llm_max_tokens * safety_margin)
        rendered = QA_PROMPT.format(context="", question="")
        self.prompt_reserve = len(self.encoding.encode(rendered)) + 50
        self.context_budget = self.total_budget - self.prompt_reserve

        # State
        self.vectorstore: Chroma | None = None
        self.sparse_retriever: BM25SparseRetriever | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None
        self.all_chunks: list[Document] = []

    # ── Helpers ──────────────────────────────────────────────────────
    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @staticmethod
    def _content_hash(doc: Document) -> str:
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    # ── Load & Index ─────────────────────────────────────────────────
    def load_and_index(self, docs_dir: Path) -> None:
        """Build both dense and sparse indexes."""
        print("\n📂  Loading and indexing documents …")
        raw_docs: list[Document] = []
        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix in {".txt", ".md"}:
                loader = TextLoader(str(fpath), encoding="utf-8")
                raw_docs.extend(loader.load())
                print(f"   ✓ {fpath.name}")

        if not raw_docs:
            raise FileNotFoundError(f"No docs in {docs_dir}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.all_chunks = splitter.split_documents(raw_docs)
        print(f"   📄 {len(self.all_chunks)} chunks created")

        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)

        self.vectorstore = Chroma.from_documents(
            documents=self.all_chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.initial_k}
        )

        self.sparse_retriever = BM25SparseRetriever.from_documents(
            self.all_chunks, k=self.initial_k
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.sparse_retriever],
            weights=[self.dense_weight, self.sparse_weight],
        )
        print("   ✓ Dense + Sparse + Ensemble indexes ready\n")

    # ── Multi-query generation ───────────────────────────────────────
    def _generate_alt_queries(self, question: str) -> list[str]:
        chain = QUERY_GEN_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "n": self.num_alt_queries})
        alts = [l.strip() for l in result.strip().split("\n") if l.strip()]
        return alts[: self.num_alt_queries]

    # ── Reranking ────────────────────────────────────────────────────
    def _rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, int]]:
        """Score each doc and return sorted (doc, score) list."""
        scored: list[tuple[Document, int]] = []
        for doc in docs:
            prompt = RERANK_PROMPT.format_messages(
                query=query, document=doc.page_content[:400],
            )
            resp = self.llm.invoke(prompt).content.strip()
            try:
                parsed = json.loads(resp)
                score = int(parsed.get("score", 1))
            except (json.JSONDecodeError, ValueError):
                match = re.search(r"\b(\d{1,2})\b", resp)
                score = int(match.group(1)) if match else 1
            scored.append((doc, max(1, min(10, score))))
        scored.sort(key=lambda x: -x[1])
        return scored

    # ── Token-budgeted context ───────────────────────────────────────
    def _budget_fit(
        self, docs: list[Document], question: str
    ) -> tuple[list[Document], int]:
        q_tokens = self._count_tokens(question)
        available = self.context_budget - q_tokens
        included: list[Document] = []
        used = 0
        for doc in docs:
            t = self._count_tokens(doc.page_content)
            if used + t <= available:
                included.append(doc)
                used += t
            else:
                break
        return included, used

    # ── Answer generation ────────────────────────────────────────────
    def _answer(self, question: str, docs: list[Document]) -> str:
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        chain = QA_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    # ── Answer quality evaluation ────────────────────────────────────
    def _evaluate_answer(self, question: str, answer: str) -> int:
        chain = QUALITY_EVAL_PROMPT | self.llm | StrOutputParser()
        resp = chain.invoke({"question": question, "answer": answer}).strip()
        try:
            return max(1, min(10, int(json.loads(resp).get("score", 5))))
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\b(\d{1,2})\b", resp)
            return int(match.group(1)) if match else 5

    # ── Relevance checker ────────────────────────────────────────────
    @staticmethod
    def _is_relevant(doc: Document, keywords: list[str]) -> bool:
        text = doc.page_content.lower()
        return any(kw.lower() in text for kw in keywords)

    # ══════════════════════════════════════════════════════════════════
    # BASIC RAG Pipeline
    # ══════════════════════════════════════════════════════════════════
    def run_basic(
        self, query: str, relevant_keywords: list[str]
    ) -> dict[str, Any]:
        """Basic RAG: single query → top-4 dense retrieval → answer."""
        assert self.vectorstore is not None
        t0 = time.perf_counter()

        docs = self.vectorstore.similarity_search(query, k=self.final_k)
        budgeted, tokens_used = self._budget_fit(docs, query)
        answer = self._answer(query, budgeted)

        latency = (time.perf_counter() - t0) * 1000

        precision = sum(
            1 for d in budgeted if self._is_relevant(d, relevant_keywords)
        ) / max(len(budgeted), 1)

        quality = self._evaluate_answer(query, answer)

        return {
            "pipeline": "Basic",
            "query": query,
            "docs_retrieved": len(docs),
            "docs_used": len(budgeted),
            "tokens_used": tokens_used,
            "precision": precision,
            "quality": quality,
            "latency_ms": latency,
            "answer": answer,
        }

    # ══════════════════════════════════════════════════════════════════
    # OPTIMIZED RAG Pipeline
    # ══════════════════════════════════════════════════════════════════
    def run_optimized(
        self, query: str, relevant_keywords: list[str]
    ) -> dict[str, Any]:
        """Full optimized pipeline: multi-query → hybrid → rerank → budget → answer."""
        assert self.ensemble_retriever is not None
        t0 = time.perf_counter()

        # Step 1: Multi-query expansion
        alt_queries = self._generate_alt_queries(query)
        all_queries = [query] + alt_queries

        # Step 2: Hybrid retrieval for each query + dedup
        seen: set[str] = set()
        candidates: list[Document] = []
        for q in all_queries:
            results = self.ensemble_retriever.invoke(q)
            for doc in results:
                h = self._content_hash(doc)
                if h not in seen:
                    seen.add(h)
                    candidates.append(doc)

        # Step 3: Rerank all unique candidates
        reranked = self._rerank(query, candidates[: min(len(candidates), 15)])
        top_docs = [doc for doc, _score in reranked[: self.initial_k]]

        # Step 4: Token budget
        budgeted, tokens_used = self._budget_fit(top_docs, query)

        # Step 5: Answer
        answer = self._answer(query, budgeted)

        latency = (time.perf_counter() - t0) * 1000

        precision = sum(
            1 for d in budgeted if self._is_relevant(d, relevant_keywords)
        ) / max(len(budgeted), 1)

        quality = self._evaluate_answer(query, answer)

        return {
            "pipeline": "Optimized",
            "query": query,
            "alt_queries": len(alt_queries),
            "unique_candidates": len(candidates),
            "docs_retrieved": len(top_docs),
            "docs_used": len(budgeted),
            "tokens_used": tokens_used,
            "precision": precision,
            "quality": quality,
            "latency_ms": latency,
            "answer": answer,
        }

    # ══════════════════════════════════════════════════════════════════
    # Benchmark
    # ══════════════════════════════════════════════════════════════════
    def run_benchmark(
        self, test_cases: list[dict[str, Any]]
    ) -> None:
        """Run both pipelines on each test case and print comparison."""
        basic_results: list[dict[str, Any]] = []
        opt_results: list[dict[str, Any]] = []

        for i, tc in enumerate(test_cases, 1):
            q = tc["query"]
            kw = tc["relevant_keywords"]

            print(f"\n{'═' * 72}")
            print(f"  BENCHMARK CASE {i}: {q}")
            print(f"{'═' * 72}")

            # Basic
            print("\n  ── BASIC RAG ──")
            basic = self.run_basic(q, kw)
            basic_results.append(basic)
            print(f"     Docs used:  {basic['docs_used']}")
            print(f"     Precision:  {basic['precision']:.2f}")
            print(f"     Quality:    {basic['quality']}/10")
            print(f"     Tokens:     {basic['tokens_used']}")
            print(f"     Latency:    {basic['latency_ms']:.0f} ms")
            preview = basic["answer"][:150].replace("\n", " ")
            print(f"     Answer:     {preview}…")

            # Optimized
            print("\n  ── OPTIMIZED RAG ──")
            opt = self.run_optimized(q, kw)
            opt_results.append(opt)
            print(f"     Alt queries:     {opt.get('alt_queries', 0)}")
            print(f"     Unique candidates: {opt.get('unique_candidates', 0)}")
            print(f"     Docs used:       {opt['docs_used']}")
            print(f"     Precision:       {opt['precision']:.2f}")
            print(f"     Quality:         {opt['quality']}/10")
            print(f"     Tokens:          {opt['tokens_used']}")
            print(f"     Latency:         {opt['latency_ms']:.0f} ms")
            preview = opt["answer"][:150].replace("\n", " ")
            print(f"     Answer:          {preview}…")

        # ── Comparison table ─────────────────────────────────────────
        print(f"\n{'═' * 80}")
        print("  📊  BENCHMARK COMPARISON: BASIC vs OPTIMIZED RAG")
        print(f"{'═' * 80}")

        header = (
            f"  {'Query':<40} {'Pipeline':<11} "
            f"{'Prec':>5} {'Qual':>5} {'Tokens':>7} {'Latency':>8}"
        )
        print(f"\n{header}")
        print(f"  {'─' * 78}")

        for b, o in zip(basic_results, opt_results):
            q_short = b["query"][:37] + "…" if len(b["query"]) > 37 else b["query"]
            print(
                f"  {q_short:<40} {'Basic':<11} "
                f"{b['precision']:>5.2f} {b['quality']:>5}/10 "
                f"{b['tokens_used']:>7} {b['latency_ms']:>7.0f}ms"
            )
            print(
                f"  {'':<40} {'Optimized':<11} "
                f"{o['precision']:>5.2f} {o['quality']:>5}/10 "
                f"{o['tokens_used']:>7} {o['latency_ms']:>7.0f}ms"
            )

        # Averages
        n = len(basic_results)
        avg_basic_prec = sum(r["precision"] for r in basic_results) / n
        avg_opt_prec = sum(r["precision"] for r in opt_results) / n
        avg_basic_qual = sum(r["quality"] for r in basic_results) / n
        avg_opt_qual = sum(r["quality"] for r in opt_results) / n
        avg_basic_tok = sum(r["tokens_used"] for r in basic_results) / n
        avg_opt_tok = sum(r["tokens_used"] for r in opt_results) / n
        avg_basic_lat = sum(r["latency_ms"] for r in basic_results) / n
        avg_opt_lat = sum(r["latency_ms"] for r in opt_results) / n

        print(f"  {'─' * 78}")
        print(
            f"  {'AVERAGES':<40} {'Basic':<11} "
            f"{avg_basic_prec:>5.2f} {avg_basic_qual:>5.1f}/10 "
            f"{avg_basic_tok:>7.0f} {avg_basic_lat:>7.0f}ms"
        )
        print(
            f"  {'':<40} {'Optimized':<11} "
            f"{avg_opt_prec:>5.2f} {avg_opt_qual:>5.1f}/10 "
            f"{avg_opt_tok:>7.0f} {avg_opt_lat:>7.0f}ms"
        )

        # Deltas
        prec_delta = avg_opt_prec - avg_basic_prec
        qual_delta = avg_opt_qual - avg_basic_qual
        lat_delta = avg_opt_lat - avg_basic_lat

        print(f"\n  📈  IMPROVEMENT DELTAS (Optimized − Basic):")
        print(f"     Precision:  {prec_delta:>+.2f}  {'↑ better' if prec_delta > 0 else '↓ worse' if prec_delta < 0 else '→ same'}")
        print(f"     Quality:    {qual_delta:>+.1f}   {'↑ better' if qual_delta > 0 else '↓ worse' if qual_delta < 0 else '→ same'}")
        print(f"     Latency:    {lat_delta:>+.0f}ms {'↑ slower (expected trade-off)' if lat_delta > 0 else '↓ faster'}")

        print(
            "\n  💡  Key Insights:"
            "\n  • Optimized RAG typically improves precision and quality"
            "\n  • The cost is higher latency (multi-query + reranking LLM calls)"
            "\n  • In production, consider async reranking and caching"
            "\n  • Use dedicated cross-encoder models for lower reranking latency"
            "\n  • Token budgeting prevents context overflow regardless of pipeline\n"
        )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the optimized RAG benchmark."""
    print("=" * 80)
    print("  Part 3 · File 15 — Optimized RAG Pipeline (All Optimizations Combined)")
    print("=" * 80)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    pipeline = OptimizedRAGPipeline(
        llm_max_tokens=8000,
        safety_margin=0.8,
        dense_weight=0.6,
        sparse_weight=0.4,
        num_alt_queries=3,
        initial_k=10,
        final_k=4,
    )
    pipeline.load_and_index(DOCS_DIR)
    pipeline.run_benchmark(BENCHMARK_CASES)

    # Cleanup
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹  Cleaned up temporary ChromaDB directory.\n")


if __name__ == "__main__":
    main()
