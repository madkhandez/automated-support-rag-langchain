"""
Part 4, File 17: Caching — Multi-Layer Cache for RAG Pipelines

Demonstrates three complementary caching strategies that dramatically reduce
latency and cost in production RAG systems:

1. **InMemoryEmbeddingCache** — Avoids redundant embedding API calls by caching
   {text_hash → embedding_vector} with LRU eviction (max 1 000 entries).

2. **QueryResultCache** — Caches {question_hash → answer} for identical
   questions with a configurable TTL (default 1 hour).  Automatically
   invalidated when new documents are ingested.

3. **Semantic cache** — If a *similar* question (cosine similarity > 0.95) has
   been asked before, the previous answer is returned instantly.

Together these layers can cut embedding costs by 50–90 % and answer repeated
questions in under 1 ms.

Key concepts:
- LRU eviction with OrderedDict
- TTL-based cache expiry
- Cosine-similarity deduplication
- Cache hit-rate monitoring
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Load environment ────────────────────────────────────────────────
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1 — Embedding Cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InMemoryEmbeddingCache:
    """
    LRU cache mapping text hashes to embedding vectors.

    When the cache exceeds *max_size* entries the least-recently-used
    item is evicted.  Hit / miss statistics are tracked automatically.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)  # FIFO = LRU end

    # ── public API ───────────────────────────────────────────────────

    def get(self, text: str) -> list[float] | None:
        """Return cached embedding or None (miss)."""
        key = self._hash(text)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)          # mark as recently used
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Insert / update an embedding and evict LRU entries if needed."""
        key = self._hash(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        self._evict_if_needed()

    def get_or_embed(
        self, text: str, embed_fn: Any,  # Callable[[str], list[float]]
    ) -> list[float]:
        """Return a cached embedding or compute, cache, and return it."""
        cached = self.get(text)
        if cached is not None:
            return cached
        embedding = embed_fn(text)
        self.put(text, embedding)
        return embedding

    def batch_get_or_embed(
        self, texts: list[str], embed_fn: Any,
    ) -> list[list[float]]:
        """
        Efficiently embed a batch — only uncached texts hit the API.
        """
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, t in enumerate(texts):
            cached = self.get(t)
            if cached is not None:
                results[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(t)

        if miss_texts:
            new_embeddings = embed_fn(miss_texts)
            for idx, text, emb in zip(miss_indices, miss_texts, new_embeddings):
                self.put(text, emb)
                results[idx] = emb

        return results  # type: ignore[return-value]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2 — Query-Result Cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class _CacheEntry:
    answer: str
    sources: list[str]
    created_at: float
    access_count: int = 0


class QueryResultCache:
    """
    TTL-based cache of (question → answer) pairs.

    Expired entries are lazily cleaned on access.  The whole cache is
    automatically invalidated when new documents are ingested so that
    stale context is never served.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _hash(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self.ttl

    # ── public API ───────────────────────────────────────────────────

    def get(self, question: str) -> dict[str, Any] | None:
        """Look up an exact-match answer.  Returns None on miss or expiry."""
        key = self._hash(question)
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self._is_expired(entry):
            del self._cache[key]
            self.misses += 1
            return None
        self.hits += 1
        entry.access_count += 1
        return {"answer": entry.answer, "sources": entry.sources, "cached": True}

    def put(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
    ) -> None:
        key = self._hash(question)
        self._cache[key] = _CacheEntry(
            answer=answer,
            sources=sources or [],
            created_at=time.time(),
        )

    def invalidate(self) -> int:
        """Clear the entire cache (e.g. after new doc ingestion). Returns old size."""
        old = len(self._cache)
        self._cache.clear()
        return old

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns count removed."""
        expired_keys = [
            k for k, v in self._cache.items() if self._is_expired(v)
        ]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3 — Semantic Cache (cosine-similarity dedup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SemanticCache:
    """
    Matches questions by *meaning* rather than exact text.

    Stores (embedding, answer) pairs and uses cosine similarity to decide
    whether a new question is semantically equivalent to a cached one.
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        threshold: float = 0.95,
        max_entries: int = 500,
    ) -> None:
        self.embeddings = embeddings
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: list[dict[str, Any]] = []  # {embedding, question, answer, sources, ts}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def get(self, question: str, threshold: float | None = None) -> dict[str, Any] | None:
        """
        Search for a semantically similar cached question.

        Returns the cached answer dict if cosine similarity ≥ threshold,
        otherwise None.
        """
        if not self._entries:
            self.misses += 1
            return None

        thr = threshold if threshold is not None else self.threshold
        q_emb = self.embeddings.embed_query(question)

        best_sim = -1.0
        best_entry: dict[str, Any] | None = None
        for entry in self._entries:
            sim = self._cosine_sim(q_emb, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= thr and best_entry is not None:
            self.hits += 1
            return {
                "answer": best_entry["answer"],
                "sources": best_entry["sources"],
                "original_question": best_entry["question"],
                "similarity": round(best_sim, 4),
                "cached": True,
            }

        self.misses += 1
        return None

    def put(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        emb = embedding or self.embeddings.embed_query(question)
        self._entries.append(
            {
                "embedding": emb,
                "question": question,
                "answer": answer,
                "sources": sources or [],
                "ts": time.time(),
            }
        )
        # Evict oldest when over capacity
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def invalidate(self) -> int:
        old = len(self._entries)
        self._entries.clear()
        return old

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "threshold": self.threshold,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified CachedRAGPipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CachedRAGPipeline:
    """
    End-to-end RAG pipeline with three caching layers.

    Cache resolution order:
      1. Exact-match query cache (fastest — pure dict lookup)
      2. Semantic cache (needs one embedding call)
      3. Full RAG pipeline (embedding + retrieval + generation)
    """

    def __init__(
        self,
        embedding_cache_size: int = 1000,
        query_cache_ttl: int = 3600,
        semantic_threshold: float = 0.95,
    ) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=0,
        )

        # Caches
        self.embedding_cache = InMemoryEmbeddingCache(max_size=embedding_cache_size)
        self.query_cache = QueryResultCache(ttl_seconds=query_cache_ttl)
        self.semantic_cache = SemanticCache(
            embeddings=self.embeddings, threshold=semantic_threshold,
        )

        # Vector store (ChromaDB for local dev)
        self._vectorstore = None
        self._total_queries: int = 0
        self._cache_sources: dict[str, int] = {
            "exact_cache": 0,
            "semantic_cache": 0,
            "full_pipeline": 0,
        }

        print("  ✅ CachedRAGPipeline initialised")
        print(f"     Embedding cache : max {embedding_cache_size} entries (LRU)")
        print(f"     Query cache     : TTL {query_cache_ttl}s")
        print(f"     Semantic cache  : threshold {semantic_threshold}")

    def _get_vectorstore(self) -> Any:
        """Lazy-init ChromaDB store."""
        if self._vectorstore is None:
            from langchain_chroma import Chroma

            persist_dir = str(
                Path(__file__).resolve().parent.parent / "chroma_db" / "part4_cache"
            )
            self._vectorstore = Chroma(
                collection_name="cached_rag",
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
        return self._vectorstore

    # ── document ingestion ───────────────────────────────────────────

    def ingest_documents(self, docs: list[Document]) -> int:
        """Ingest documents and invalidate answer caches."""
        vs = self._get_vectorstore()
        vs.add_documents(docs)

        # Invalidate answer caches — retrieved context may have changed
        n_query = self.query_cache.invalidate()
        n_sem = self.semantic_cache.invalidate()
        print(f"  🗑  Invalidated {n_query} query-cache + {n_sem} semantic-cache entries")
        return len(docs)

    # ── query ────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        k: int = 4,
        semantic_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Answer a question using the three-layer cache hierarchy.

        Returns a dict with *answer*, *sources*, *cache_layer*, and *latency_ms*.
        """
        self._total_queries += 1
        t0 = time.perf_counter()

        # Layer 1: exact-match query cache
        exact = self.query_cache.get(question)
        if exact is not None:
            self._cache_sources["exact_cache"] += 1
            return {
                **exact,
                "cache_layer": "exact_cache",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # Layer 2: semantic cache
        thr = semantic_threshold if semantic_threshold is not None else self.semantic_cache.threshold
        sem = self.semantic_cache.get(question, threshold=thr)
        if sem is not None:
            # Also populate exact cache for next identical hit
            self.query_cache.put(question, sem["answer"], sem["sources"])
            self._cache_sources["semantic_cache"] += 1
            return {
                **sem,
                "cache_layer": "semantic_cache",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # Layer 3: full RAG pipeline
        answer_data = self._full_pipeline(question, k)
        self._cache_sources["full_pipeline"] += 1
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # Populate caches
        self.query_cache.put(question, answer_data["answer"], answer_data["sources"])
        self.semantic_cache.put(question, answer_data["answer"], answer_data["sources"])

        return {
            **answer_data,
            "cached": False,
            "cache_layer": "full_pipeline",
            "latency_ms": elapsed,
        }

    def _full_pipeline(self, question: str, k: int) -> dict[str, Any]:
        """Run the full retrieve → generate pipeline."""
        vs = self._get_vectorstore()

        # Retrieve
        docs = vs.similarity_search(question, k=k)
        sources = [d.metadata.get("source", "unknown") for d in docs]
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Generate
        prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": sources,
        }

    # ── dashboard ────────────────────────────────────────────────────

    def print_cache_dashboard(self) -> None:
        """Pretty-print cache statistics."""
        print("\n" + "─" * 60)
        print("  📊  Cache Dashboard")
        print("─" * 60)
        print(f"  Total queries: {self._total_queries}")
        print()

        for label, cache in [
            ("Embedding Cache", self.embedding_cache),
            ("Query Cache    ", self.query_cache),
            ("Semantic Cache ", self.semantic_cache),
        ]:
            s = cache.stats()
            print(f"  {label}  │  size={s.get('size', s.get('entries', '?')):<5}  "
                  f"hits={s['hits']:<4}  misses={s['misses']:<4}  "
                  f"rate={s['hit_rate']}")

        print()
        print("  Queries resolved by layer:")
        for layer, count in self._cache_sources.items():
            pct = count / self._total_queries * 100 if self._total_queries else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {layer:<16} {count:>4}  ({pct:5.1f}%)  {bar}")
        print("─" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _sample_docs() -> list[Document]:
    """Build a small set of sample documents."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs: list[Document] = []

    for path in sorted(docs_dir.glob("*.txt")) + sorted(docs_dir.glob("*.md")):
        loader = TextLoader(str(path), encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        all_docs.extend(chunks)
        print(f"  📄 {path.name} → {len(chunks)} chunks")

    if not all_docs:
        all_docs = [
            Document(page_content="RAG combines retrieval with generation to answer questions.",
                     metadata={"source": "synthetic"}),
            Document(page_content="Caching is essential in production to reduce latency and cost.",
                     metadata={"source": "synthetic"}),
            Document(page_content="LangChain provides tools for building LLM-powered applications.",
                     metadata={"source": "synthetic"}),
            Document(page_content="Vector databases store embeddings for similarity search.",
                     metadata={"source": "synthetic"}),
        ]
    return all_docs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Run a full caching demo showing hit rates across all three layers."""

    print("=" * 70)
    print("  Part 4 · File 17 — Multi-Layer RAG Caching")
    print("=" * 70)

    # ── 1. Initialise pipeline ───────────────────────────────────────
    print("\n🔧 Step 1: Initialising CachedRAGPipeline …")
    pipeline = CachedRAGPipeline(
        embedding_cache_size=1000,
        query_cache_ttl=3600,
        semantic_threshold=0.95,
    )

    # ── 2. Ingest documents ──────────────────────────────────────────
    print("\n📥 Step 2: Ingesting sample documents …")
    docs = _sample_docs()
    n = pipeline.ingest_documents(docs)
    print(f"  → Ingested {n} documents")

    # ── 3. Demo: embedding cache ─────────────────────────────────────
    print("\n⚡ Step 3: Embedding cache demo")
    cache = pipeline.embedding_cache
    embed_fn = pipeline.embeddings.embed_query

    texts = [
        "What is RAG?",
        "How does caching work?",
        "What is RAG?",           # exact repeat
        "How does caching work?",  # exact repeat
        "Tell me about LangChain",
    ]
    for text in texts:
        _ = cache.get_or_embed(text, embed_fn)
    print(f"  Embedding cache stats after 5 calls (2 repeats):")
    for k, v in cache.stats().items():
        print(f"    {k}: {v}")

    # ── 4. Demo: full queries showing cache layers ───────────────────
    print("\n🔍 Step 4: Query pipeline with caching")
    questions = [
        "What is RAG and how does retrieval augmented generation work?",
        "What is RAG and how does retrieval augmented generation work?",  # exact repeat → Layer 1
        "Explain RAG — retrieval augmented generation",                    # semantic match → Layer 2
        "How does caching improve performance?",                           # new → Layer 3
        "How does caching improve performance?",                           # exact repeat → Layer 1
    ]

    for i, q in enumerate(questions, 1):
        result = pipeline.query(q)
        layer = result["cache_layer"]
        latency = result["latency_ms"]
        is_cached = result.get("cached", False)
        print(f"\n  Q{i}: "{q[:60]}…"")
        print(f"      Layer: {layer}  │  Cached: {is_cached}  │  Latency: {latency:.1f} ms")
        print(f"      Answer: {result['answer'][:120]}…")

    # ── 5. Cache invalidation demo ───────────────────────────────────
    print("\n\n🗑  Step 5: Cache invalidation on new document ingestion")
    new_doc = Document(
        page_content="New information about advanced caching patterns in 2025.",
        metadata={"source": "new_doc"},
    )
    pipeline.ingest_documents([new_doc])
    result_after = pipeline.query("How does caching improve performance?")
    print(f"  After invalidation → Layer: {result_after['cache_layer']}  "
          f"(should be full_pipeline)")

    # ── 6. Dashboard ─────────────────────────────────────────────────
    print("\n📊 Step 6: Final cache dashboard")
    pipeline.print_cache_dashboard()

    print("\n" + "=" * 70)
    print("  ✅ Caching demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
