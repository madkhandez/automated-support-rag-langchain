"""
Part 1 — Vector Stores with ChromaDB
======================================
Manage ChromaDB vector store: create collections, index documents,
similarity search with scores, metadata filtering, and collection stats.
"""

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION = os.environ.get("COLLECTION_NAME", "production_rag_docs")


class VectorStoreManager:
    """Full-featured ChromaDB vector store management."""

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        self.persist_dir = persist_dir or CHROMA_DIR
        self.collection_name = collection_name or COLLECTION
        self.embeddings = OpenAIEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self._Chroma = Chroma
        self.vectorstore = None

    # ── Create / load collection ────────────────────────────────────
    def get_or_create(self):
        """Get existing collection or create a new one."""
        self.vectorstore = self._Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"  📂 Vector store ready: {self.collection_name}")
        print(f"     Persist directory: {self.persist_dir}")
        return self.vectorstore

    # ── Index documents ─────────────────────────────────────────────
    def index_documents(self, documents: list) -> None:
        """Embed and store documents in one pipeline."""
        print(f"\n  📥 Indexing {len(documents)} documents...")
        self.vectorstore = self._Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        print(f"  ✅ Indexed {len(documents)} documents into '{self.collection_name}'")

    # ── Similarity search ───────────────────────────────────────────
    def similarity_search(self, query: str, k: int = 4) -> list:
        """Search for similar documents, returning Document objects."""
        if not self.vectorstore:
            self.get_or_create()
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def similarity_search_with_score(self, query: str, k: int = 4) -> list:
        """Search returning (Document, score) tuples."""
        if not self.vectorstore:
            self.get_or_create()
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    # ── Metadata-filtered search ────────────────────────────────────
    def filtered_search(self, query: str, filter_dict: dict, k: int = 4) -> list:
        """Search with metadata filters."""
        if not self.vectorstore:
            self.get_or_create()
        results = self.vectorstore.similarity_search(
            query, k=k, filter=filter_dict
        )
        return results

    # ── Collection stats ────────────────────────────────────────────
    def collection_stats(self) -> dict:
        """Get collection statistics."""
        if not self.vectorstore:
            self.get_or_create()

        collection = self.vectorstore._collection
        count = collection.count()

        # Calculate disk size
        disk_size = 0
        if os.path.exists(self.persist_dir):
            for dirpath, _, filenames in os.walk(self.persist_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    disk_size += os.path.getsize(fp)

        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_dir,
            "disk_size_bytes": disk_size,
            "disk_size_mb": round(disk_size / (1024 * 1024), 2),
        }
        return stats

    # ── Delete / reset ──────────────────────────────────────────────
    def delete_collection(self):
        """Delete the current collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print(f"  🗑️  Deleted collection: {self.collection_name}")
            self.vectorstore = None

    def reset_collection(self):
        """Delete and recreate the collection."""
        self.delete_collection()
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            print(f"  🗑️  Removed persist directory: {self.persist_dir}")
        self.get_or_create()
        print(f"  🔄 Collection reset: {self.collection_name}")


def main():
    """Demonstrate all VectorStoreManager capabilities."""
    print("🗄️  VECTOR STORES — ChromaDB")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in .env to run this module.")
        return

    from langchain_core.documents import Document

    # Use a temporary directory for this demo
    demo_dir = os.path.join(os.path.dirname(__file__), ".demo_chroma_db")
    manager = VectorStoreManager(
        persist_dir=demo_dir,
        collection_name="demo_collection"
    )

    # ── 1. Create sample documents ──────────────────────────────────
    print("\n📝 Step 1: Creating sample documents...")
    docs = [
        Document(
            page_content="ChromaDB is an open-source vector database for AI applications. "
                         "It runs locally without a server and is ideal for development.",
            metadata={"source": "tech_docs", "topic": "databases"}
        ),
        Document(
            page_content="PGVector extends PostgreSQL with vector similarity search. "
                         "It supports HNSW and IVFFlat indexing algorithms.",
            metadata={"source": "tech_docs", "topic": "databases"}
        ),
        Document(
            page_content="Employees are entitled to 15 days of paid annual leave per year. "
                         "Leave requests must be submitted 14 days in advance.",
            metadata={"source": "company_policy", "topic": "hr"}
        ),
        Document(
            page_content="RAG combines retrieval with generation to answer questions "
                         "grounded in specific documents, reducing hallucination.",
            metadata={"source": "langchain_docs", "topic": "rag"}
        ),
        Document(
            page_content="LangChain provides over 100 document loaders for ingesting "
                         "data from PDFs, web pages, databases, and APIs.",
            metadata={"source": "langchain_docs", "topic": "langchain"}
        ),
    ]

    # ── 2. Index documents ──────────────────────────────────────────
    print("\n📥 Step 2: Indexing documents...")
    manager.index_documents(docs)

    # ── 3. Similarity search ────────────────────────────────────────
    print("\n🔍 Step 3: Similarity search")
    query = "What vector databases are available?"
    results = manager.similarity_search(query, k=3)
    print(f"  Query: '{query}'")
    print(f"  Results ({len(results)} docs):")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:80]
        src = doc.metadata.get("source", "unknown")
        print(f"    [{i}] ({src}) {preview}...")

    # ── 4. Search with scores ───────────────────────────────────────
    print("\n📊 Step 4: Similarity search WITH scores")
    print("  Score interpretation (L2 distance):")
    print("    0.0 = identical | 1.0 = orthogonal | 2.0 = opposite")
    print()

    scored = manager.similarity_search_with_score(query, k=4)
    for i, (doc, score) in enumerate(scored, 1):
        preview = doc.page_content[:60]
        bar = "█" * max(1, int((2.0 - score) * 15))
        print(f"    [{i}] Score: {score:.4f} {bar} {preview}...")

    # ── 5. Metadata-filtered search ─────────────────────────────────
    print("\n🏷️  Step 5: Metadata-filtered search")
    query2 = "How do I manage my leave?"
    print(f"  Query: '{query2}'")

    # Without filter
    all_results = manager.similarity_search(query2, k=3)
    print(f"\n  Without filter ({len(all_results)} results):")
    for doc in all_results:
        print(f"    • [{doc.metadata.get('source')}] {doc.page_content[:60]}...")

    # With filter: only company_policy
    filtered = manager.filtered_search(
        query2, filter_dict={"source": "company_policy"}, k=3
    )
    print(f"\n  With filter source='company_policy' ({len(filtered)} results):")
    for doc in filtered:
        print(f"    • [{doc.metadata.get('source')}] {doc.page_content[:60]}...")

    # ── 6. Collection stats ─────────────────────────────────────────
    print("\n📈 Step 6: Collection statistics")
    stats = manager.collection_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # ── 7. Cleanup ──────────────────────────────────────────────────
    print("\n🧹 Step 7: Cleanup")
    manager.reset_collection()
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
        print(f"  🗑️  Removed demo directory: {demo_dir}")

    print("\n" + "=" * 70)
    print("✅ Vector Stores demo complete!")


if __name__ == "__main__":
    main()
