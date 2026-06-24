"""
Part 5 — Vector Store Factory: Environment-Aware Vector Store Management.

Automatically selects the correct vector-store backend based on the
``ENVIRONMENT`` env var:

  • **development** → local ChromaDB (zero-config, works offline)
  • **production**  → Supabase pgvector (falls back to ChromaDB if creds
    are missing)

Exposes ``add_documents`` and ``similarity_search`` convenience wrappers so
the rest of the codebase never touches the underlying store directly.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

# ── Load environment variables ───────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class VectorStoreFactory:
    """Singleton factory that provisions and caches the active vector store.

    Usage::

        vs = VectorStoreFactory()
        store = vs.get_vector_store()           # ChromaDB or Supabase
        vs.add_documents([doc1, doc2])
        results = vs.similarity_search("query", k=4)
    """

    _instance: Optional["VectorStoreFactory"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ────────────────────────────────────────────────────
    def __new__(cls) -> "VectorStoreFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.environment: str = os.getenv("ENVIRONMENT", "development").lower()
        self.collection_name: str = os.getenv("COLLECTION_NAME", "production_rag_docs")
        self.chroma_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self._store: Any = None
        self._backend_name: str = ""

        print(f"🏭 VectorStoreFactory  env={self.environment}  "
              f"collection={self.collection_name}")

    # ── Embeddings (lazy) ────────────────────────────────────────────
    def _get_embeddings(self):
        """Retrieve embeddings from the LLMFactory singleton."""
        from part5_production.core.llm import LLMFactory
        return LLMFactory().get_embeddings()

    # ── ChromaDB ─────────────────────────────────────────────────────
    def _create_chroma(self):
        """Create a local ChromaDB vector store (offline-capable)."""
        from langchain_chroma import Chroma

        persist_dir = str(Path(self.chroma_dir).resolve())
        os.makedirs(persist_dir, exist_ok=True)

        store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=persist_dir,
        )
        self._backend_name = "chromadb"
        print(f"   ✅ ChromaDB ready  dir={persist_dir}")
        return store

    # ── Supabase / pgvector ──────────────────────────────────────────
    def _create_supabase(self):
        """Create a Supabase pgvector vector store.

        Falls back to ChromaDB if required environment variables are missing
        or if the ``supabase`` package is not installed.
        """
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

        if not supabase_url or not supabase_key:
            print("   ⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY not set → "
                  "falling back to ChromaDB")
            return self._create_chroma()

        try:
            from supabase import create_client
            from langchain_community.vectorstores import SupabaseVectorStore

            client = create_client(supabase_url, supabase_key)
            store = SupabaseVectorStore(
                client=client,
                embedding=self._get_embeddings(),
                table_name="documents",
                query_name="match_documents",
            )
            self._backend_name = "supabase"
            print(f"   ✅ Supabase vector store connected  url={supabase_url[:40]}…")
            return store

        except ImportError:
            print("   ⚠️  supabase / langchain_community not installed → "
                  "falling back to ChromaDB")
            return self._create_chroma()
        except Exception as exc:
            print(f"   ⚠️  Supabase init failed ({exc}) → falling back to ChromaDB")
            return self._create_chroma()

    # ── Public API ───────────────────────────────────────────────────
    def get_vector_store(self):
        """Return the active vector store (created on first call)."""
        if self._store is None:
            print("🔧 Provisioning vector store …")
            if self.environment == "production":
                self._store = self._create_supabase()
            else:
                self._store = self._create_chroma()
        return self._store

    @property
    def backend_name(self) -> str:
        """Name of the current backend (``'chromadb'`` or ``'supabase'``)."""
        if not self._backend_name:
            self.get_vector_store()  # ensure initialised
        return self._backend_name

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Index a list of LangChain Document objects.

        Returns:
            List of document IDs assigned by the store.
        """
        store = self.get_vector_store()
        print(f"📥 Indexing {len(documents)} document(s) into {self.backend_name} …")
        ids = store.add_documents(documents)
        print(f"   ✅ Indexed {len(ids)} chunk(s)")
        return ids

    def similarity_search(
        self, query: str, k: int = 4
    ) -> list[Document]:
        """Run a similarity search and return the top-k documents."""
        store = self.get_vector_store()
        results = store.similarity_search(query, k=k)
        print(f"🔍 similarity_search  k={k}  results={len(results)}  "
              f"backend={self.backend_name}")
        return results

    def test_connection(self) -> bool:
        """Verify the vector store is reachable."""
        try:
            store = self.get_vector_store()
            # A lightweight probe — search for an empty-ish string
            store.similarity_search("health check", k=1)
            print(f"✅ Vector store ({self.backend_name}) connection healthy")
            return True
        except Exception as exc:
            print(f"❌ Vector store connection failed: {exc}")
            return False


# ── Standalone entrypoint ────────────────────────────────────────────
def main() -> None:
    """Demonstrate VectorStoreFactory usage."""
    print("=" * 60)
    print("Part 5 · Vector Store Factory Demo")
    print("=" * 60)

    factory = VectorStoreFactory()

    # 1. Provision the store
    print("\n── Provisioning ──")
    store = factory.get_vector_store()
    print(f"   Backend: {factory.backend_name}")
    print(f"   Store type: {type(store).__name__}")

    # 2. Add sample documents
    print("\n── Adding documents ──")
    sample_docs = [
        Document(
            page_content="ChromaDB is a lightweight vector database for local development.",
            metadata={"source": "demo", "topic": "databases"},
        ),
        Document(
            page_content="Supabase provides hosted Postgres with pgvector support.",
            metadata={"source": "demo", "topic": "databases"},
        ),
    ]
    ids = factory.add_documents(sample_docs)
    print(f"   IDs: {ids}")

    # 3. Search
    print("\n── Similarity search ──")
    results = factory.similarity_search("vector database for production", k=2)
    for i, doc in enumerate(results, 1):
        print(f"   [{i}] {doc.page_content[:80]}…")

    # 4. Health check
    print("\n── Health check ──")
    ok = factory.test_connection()
    print(f"   Healthy: {ok}")

    print("\n✅ Vector Store Factory demo complete.")


if __name__ == "__main__":
    main()
