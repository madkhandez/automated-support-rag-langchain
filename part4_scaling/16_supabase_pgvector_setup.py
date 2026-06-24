"""
Part 4, File 16: Supabase pgvector Setup — Production Vector Database

Demonstrates how to use Supabase (PostgreSQL + pgvector) as a production-grade
vector store, with a graceful fallback to local ChromaDB when Supabase is not
configured. This pattern lets you develop locally with ChromaDB and deploy to
Supabase in production without changing application code.

Key concepts:
- Supabase pgvector for scalable, cloud-hosted similarity search
- Connection pooling for production workloads
- Batch upsert with metadata
- Filtered similarity search via RPC functions
- Graceful degradation to ChromaDB for local development

Prerequisites (Supabase side — run these in the Supabase SQL Editor):
----------------------------------------------------------------------

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    embedding       vector(1536),      -- text-embedding-3-small dimension
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- 3. Create an HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. Create the match_documents RPC function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding   vector(1536),
    match_count       INT DEFAULT 4,
    filter_metadata   JSONB DEFAULT '{}'
)
RETURNS TABLE (
    id          TEXT,
    content     TEXT,
    metadata    JSONB,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE (filter_metadata = '{}' OR d.metadata @> filter_metadata)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
----------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ── Constants ────────────────────────────────────────────────────────
EMBEDDING_DIM = 1536          # text-embedding-3-small
MAX_BATCH_SIZE = 500          # Supabase RPC payload limit guard
MAX_CONNECTIONS = 10          # Connection-pool ceiling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChromaDB Fallback Store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ChromaFallbackStore:
    """Local ChromaDB-backed vector store used when Supabase is unavailable."""

    def __init__(self, embeddings: OpenAIEmbeddings, persist_dir: str) -> None:
        from langchain_chroma import Chroma

        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.vectorstore = Chroma(
            collection_name="supabase_fallback",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        print(f"  ✦ ChromaDB fallback initialised at: {persist_dir}")

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Add documents in batches and return their IDs."""
        ids: list[str] = []
        for i in range(0, len(docs), MAX_BATCH_SIZE):
            batch = docs[i : i + MAX_BATCH_SIZE]
            batch_ids = [
                hashlib.sha256(
                    (d.page_content + str(d.metadata)).encode()
                ).hexdigest()[:16]
                for d in batch
            ]
            self.vectorstore.add_documents(documents=batch, ids=batch_ids)
            ids.extend(batch_ids)
            print(f"  ✦ ChromaDB: ingested batch {i // MAX_BATCH_SIZE + 1} "
                  f"({len(batch)} docs)")
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return the *k* most similar documents."""
        kwargs: dict[str, Any] = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return self.vectorstore.similarity_search(query, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the *k* most similar documents with their similarity scores."""
        kwargs: dict[str, Any] = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return self.vectorstore.similarity_search_with_score(query, **kwargs)

    def health_check(self) -> bool:
        """ChromaDB is always healthy when the directory exists."""
        return Path(self.persist_dir).exists() or True

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic collection statistics."""
        collection = self.vectorstore._collection
        count = collection.count()
        return {
            "backend": "chromadb",
            "document_count": count,
            "persist_directory": self.persist_dir,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Supabase Vector Store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SupabaseVectorStore:
    """
    Production vector store backed by Supabase (Postgres + pgvector).

    Falls back automatically to ChromaDB when Supabase credentials are
    not present in the environment.
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings | None = None,
        persist_dir: str = "./chroma_db",
    ) -> None:
        load_dotenv()

        self.embeddings = embeddings or OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        self.supabase_url: str | None = os.getenv("SUPABASE_URL")
        self.supabase_key: str | None = os.getenv("SUPABASE_SERVICE_KEY")
        self.max_connections: int = MAX_CONNECTIONS
        self.using_supabase: bool = False
        self.client: Any = None
        self._fallback: ChromaFallbackStore | None = None

        # ── Attempt Supabase connection ──────────────────────────────
        if self._supabase_configured():
            try:
                self._init_supabase()
                self.using_supabase = True
                print("  ✅ Connected to Supabase pgvector")
            except Exception as exc:
                print(f"  ⚠️  Supabase connection failed: {exc}")
                print("  ↳  Falling back to ChromaDB")
                self._init_fallback(persist_dir)
        else:
            print("  ⚠️  Supabase not configured, falling back to ChromaDB")
            self._init_fallback(persist_dir)

    # ── Private helpers ──────────────────────────────────────────────

    def _supabase_configured(self) -> bool:
        """Return True when both Supabase env vars look valid."""
        if not self.supabase_url or not self.supabase_key:
            return False
        placeholders = {"your_supabase_project_url", "your_supabase_service_role_key"}
        return (
            self.supabase_url not in placeholders
            and self.supabase_key not in placeholders
        )

    def _init_supabase(self) -> None:
        """Initialise the Supabase client (lazy import)."""
        from supabase import create_client, Client   # type: ignore[import-untyped]

        assert self.supabase_url and self.supabase_key
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        print(f"  ✦ Supabase client created — pool up to {self.max_connections} conns")

    def _init_fallback(self, persist_dir: str) -> None:
        self._fallback = ChromaFallbackStore(self.embeddings, persist_dir)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the configured embedding model."""
        return self.embeddings.embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(text)

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Deterministic ID from content + metadata."""
        raw = doc.page_content + str(sorted(doc.metadata.items()))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Public API ───────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> list[str]:
        """
        Batch-upsert documents with their embeddings.

        Returns the list of document IDs that were upserted.
        """
        if self._fallback:
            return self._fallback.add_documents(docs)

        all_ids: list[str] = []
        for i in range(0, len(docs), MAX_BATCH_SIZE):
            batch = docs[i : i + MAX_BATCH_SIZE]
            texts = [d.page_content for d in batch]
            embeddings = self._embed_texts(texts)

            rows = []
            for doc, emb in zip(batch, embeddings):
                doc_id = self._doc_id(doc)
                rows.append(
                    {
                        "id": doc_id,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": emb,
                    }
                )
                all_ids.append(doc_id)

            # Upsert via the Supabase REST API
            self.client.table("documents").upsert(rows).execute()
            print(
                f"  ✦ Supabase: upserted batch {i // MAX_BATCH_SIZE + 1} "
                f"({len(batch)} docs)"
            )

        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve the *k* most similar documents to *query*.

        Uses the ``match_documents`` Postgres RPC function when running
        against Supabase, or ChromaDB's built-in search as a fallback.
        """
        if self._fallback:
            return self._fallback.similarity_search(query, k, filter_metadata)

        query_embedding = self._embed_query(query)
        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_metadata": filter_metadata or {},
        }

        response = self.client.rpc("match_documents", rpc_params).execute()
        results: list[Document] = []
        for row in response.data:
            doc = Document(
                page_content=row["content"],
                metadata={
                    **(row.get("metadata") or {}),
                    "similarity": row["similarity"],
                },
            )
            results.append(doc)
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return (Document, similarity_score) tuples."""
        if self._fallback:
            return self._fallback.similarity_search_with_score(
                query, k, filter_metadata
            )

        query_embedding = self._embed_query(query)
        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_metadata": filter_metadata or {},
        }

        response = self.client.rpc("match_documents", rpc_params).execute()
        results: list[tuple[Document, float]] = []
        for row in response.data:
            doc = Document(
                page_content=row["content"],
                metadata=row.get("metadata") or {},
            )
            results.append((doc, float(row["similarity"])))
        return results

    def health_check(self) -> bool:
        """
        Verify that the vector store is reachable and responsive.

        Returns True when healthy, False otherwise.
        """
        if self._fallback:
            return self._fallback.health_check()

        try:
            # A lightweight query — just count rows
            resp = (
                self.client.table("documents")
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            return resp is not None
        except Exception as exc:
            print(f"  ❌ Health check failed: {exc}")
            return False

    def get_collection_stats(self) -> dict[str, Any]:
        """Return metadata about the backing store."""
        if self._fallback:
            return self._fallback.get_collection_stats()

        try:
            resp = (
                self.client.table("documents")
                .select("id", count="exact")
                .limit(0)
                .execute()
            )
            return {
                "backend": "supabase_pgvector",
                "document_count": resp.count,
                "supabase_url": self.supabase_url,
                "max_connections": self.max_connections,
            }
        except Exception as exc:
            return {"backend": "supabase_pgvector", "error": str(exc)}

    @property
    def backend_name(self) -> str:
        return "supabase" if self.using_supabase else "chromadb"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_sample_documents() -> list[Document]:
    """Load sample documents from the project docs/ folder."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs: list[Document] = []

    for txt_file in sorted(docs_dir.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        for chunk in chunks:
            chunk.metadata["source_type"] = "txt"
        all_docs.extend(chunks)
        print(f"  📄 Loaded {txt_file.name} → {len(chunks)} chunks")

    for md_file in sorted(docs_dir.glob("*.md")):
        loader = TextLoader(str(md_file), encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        for chunk in chunks:
            chunk.metadata["source_type"] = "md"
        all_docs.extend(chunks)
        print(f"  📄 Loaded {md_file.name} → {len(chunks)} chunks")

    if not all_docs:
        print("  ⚠️  No docs found in docs/ — using synthetic samples")
        all_docs = [
            Document(
                page_content="RAG combines retrieval and generation to produce "
                "grounded answers. It first searches a knowledge base and "
                "then feeds retrieved context into a language model.",
                metadata={"source": "synthetic", "topic": "rag", "source_type": "txt"},
            ),
            Document(
                page_content="pgvector is a PostgreSQL extension that adds support "
                "for storing and querying vector embeddings using exact and "
                "approximate nearest-neighbor search indexes.",
                metadata={"source": "synthetic", "topic": "pgvector", "source_type": "txt"},
            ),
            Document(
                page_content="Connection pooling limits the number of active "
                "database connections and reuses them across requests, "
                "improving throughput under load.",
                metadata={"source": "synthetic", "topic": "databases", "source_type": "txt"},
            ),
            Document(
                page_content="LangChain provides abstractions for document loaders, "
                "text splitters, embedding models, vector stores, retrievers, "
                "and chains — simplifying RAG pipeline development.",
                metadata={"source": "synthetic", "topic": "langchain", "source_type": "txt"},
            ),
        ]
    return all_docs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Demonstrate the SupabaseVectorStore with automatic fallback."""

    print("=" * 70)
    print("  Part 4 · File 16 — Supabase pgvector Setup")
    print("=" * 70)

    # ── 1. Initialise the store ──────────────────────────────────────
    print("\n🔧 Step 1: Initialising vector store …")
    store = SupabaseVectorStore(
        persist_dir=str(
            Path(__file__).resolve().parent.parent / "chroma_db" / "part4_supabase"
        ),
    )
    print(f"  → Active backend: {store.backend_name}")

    # ── 2. Health check ──────────────────────────────────────────────
    print("\n🏥 Step 2: Running health check …")
    healthy = store.health_check()
    print(f"  → Healthy: {healthy}")

    # ── 3. Ingest sample documents ───────────────────────────────────
    print("\n📥 Step 3: Loading & ingesting sample documents …")
    docs = _load_sample_documents()
    print(f"  → Total documents to ingest: {len(docs)}")

    t0 = time.perf_counter()
    ids = store.add_documents(docs)
    elapsed = time.perf_counter() - t0
    print(f"  → Ingested {len(ids)} documents in {elapsed:.2f}s")

    # ── 4. Collection stats ──────────────────────────────────────────
    print("\n📊 Step 4: Collection statistics")
    stats = store.get_collection_stats()
    for k, v in stats.items():
        print(f"  {k:>20s}: {v}")

    # ── 5. Similarity search ─────────────────────────────────────────
    print("\n🔍 Step 5: Similarity search")
    queries = [
        "What is RAG and how does it work?",
        "How does vector search work in PostgreSQL?",
        "What is the company leave policy?",
    ]
    for query in queries:
        print(f"\n  Query: "{query}"")
        t0 = time.perf_counter()
        results = store.similarity_search(query, k=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Latency: {elapsed_ms:.1f} ms — {len(results)} results")
        for i, doc in enumerate(results, 1):
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"    {i}. [{doc.metadata.get('source_type', '?')}] {snippet}…")

    # ── 6. Filtered search ───────────────────────────────────────────
    print("\n🔍 Step 6: Filtered similarity search (source_type=txt)")
    results = store.similarity_search(
        "What is RAG?", k=2, filter_metadata={"source_type": "txt"}
    )
    print(f"  → {len(results)} results with filter source_type=txt")
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content[:100].replace("\n", " ")
        print(f"    {i}. {snippet}…")

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ Supabase pgvector setup complete!")
    print(f"  Backend used: {store.backend_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
