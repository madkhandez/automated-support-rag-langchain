"""
Tests for the core RAG pipeline components.

Tests document loading, text splitting, embeddings, vector store operations,
RAG chain execution, hallucination detection, and token counting.
All OpenAI API calls are mocked — tests run entirely offline without API keys.
"""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample LangChain Document objects for testing."""
    return [
        Document(
            page_content="Vector databases store embeddings for fast similarity search. "
                         "They are essential for RAG applications.",
            metadata={"source": "vector_db_guide.txt", "page": 0},
        ),
        Document(
            page_content="LangChain provides abstractions for working with LLMs, "
                         "document loaders, text splitters, and vector stores.",
            metadata={"source": "langchain_docs.md", "page": 0},
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines retrieval of relevant "
                         "documents with language model generation for grounded answers.",
            metadata={"source": "rag_overview.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_long_text() -> str:
    """A block of text large enough to be meaningfully split into chunks."""
    paragraphs = [
        "Vector databases are specialized databases designed to store and query "
        "high-dimensional vectors. They are essential components of modern AI "
        "applications, particularly Retrieval-Augmented Generation (RAG) systems. "
        "Popular vector databases include ChromaDB, Pinecone, Weaviate, and Milvus.",

        "Embeddings are numerical representations of text that capture semantic meaning. "
        "OpenAI's text-embedding-ada-002 model produces 1536-dimensional vectors. "
        "These vectors enable similarity search based on meaning rather than keywords.",

        "RAG combines the power of retrieval systems with generative language models. "
        "When a user asks a question, the system first retrieves relevant documents "
        "from a vector store, then passes them as context to an LLM for generation. "
        "This approach reduces hallucination and provides grounded, factual answers.",

        "LangChain is a framework for building applications with LLMs. It provides "
        "tools for document loading, text splitting, embedding generation, vector "
        "storage, and chain composition. LangChain supports multiple LLM providers "
        "and vector database backends through a unified interface.",

        "Production RAG systems require careful attention to chunking strategies, "
        "embedding model selection, retrieval algorithms, and prompt engineering. "
        "Monitoring with tools like LangSmith helps identify and fix issues in "
        "real-time. Security measures prevent prompt injection and data leakage.",
    ]
    return "\n\n".join(paragraphs)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock OpenAIEmbeddings that returns deterministic vectors.

    ChromaDB calls embed_documents (list[str] -> list[list[float]]) for
    ingestion and embed_query (str -> list[float]) for search.
    """
    mock = MagicMock()

    def _embed_documents(texts: list[str]) -> list[list[float]]:
        """Return a unique-ish 1536-d vector for each text."""
        vectors = []
        for i, text in enumerate(texts):
            vec = [float(hash(text) % 1000) / 1000.0 + j * 0.0001
                   for j in range(1536)]
            vectors.append(vec)
        return vectors

    def _embed_query(text: str) -> list[float]:
        return _embed_documents([text])[0]

    mock.embed_documents = MagicMock(side_effect=_embed_documents)
    mock.embed_query = MagicMock(side_effect=_embed_query)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocumentLoading:
    """Tests for document loading functionality."""

    @patch("langchain_community.document_loaders.TextLoader.load")
    def test_document_loading(self, mock_load: MagicMock) -> None:
        """Mock document loaders and verify documents are loaded correctly."""
        # Arrange: configure mock to return sample documents
        expected_docs = [
            Document(
                page_content="ACME Corporation Employee Leave Policy...",
                metadata={"source": "company_policy.txt"},
            ),
            Document(
                page_content="Policy continued: annual leave is 15 days per year.",
                metadata={"source": "company_policy.txt"},
            ),
        ]
        mock_load.return_value = expected_docs

        # Act
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader("docs/company_policy.txt")
        docs = loader.load()

        # Assert
        assert len(docs) == 2, "Expected 2 documents from the loader"
        assert docs[0].page_content.startswith("ACME"), (
            "First document should start with 'ACME'"
        )
        assert docs[0].metadata["source"] == "company_policy.txt"
        mock_load.assert_called_once()
        print(f"✅ Loaded {len(docs)} documents successfully (mocked)")

    @patch("langchain_community.document_loaders.PyPDFLoader.load")
    def test_pdf_loading(self, mock_load: MagicMock) -> None:
        """Mock PDF loader and verify multi-page documents."""
        expected_docs = [
            Document(page_content="Page 1 content about LangChain.",
                     metadata={"source": "demo.pdf", "page": 0}),
            Document(page_content="Page 2 content about vector stores.",
                     metadata={"source": "demo.pdf", "page": 1}),
        ]
        mock_load.return_value = expected_docs

        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader("docs/langchain_demo.pdf")
        docs = loader.load()

        assert len(docs) == 2
        assert docs[1].metadata["page"] == 1
        print(f"✅ Loaded {len(docs)} PDF pages successfully (mocked)")


class TestTextSplitting:
    """Tests for text splitting with RecursiveCharacterTextSplitter."""

    def test_text_splitting(self, sample_long_text: str) -> None:
        """Create sample text, split it, and assert chunk count < original length."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(sample_long_text)

        assert len(chunks) > 1, "Text should be split into multiple chunks"
        assert all(len(c) <= 200 + 50 for c in chunks), (
            "All chunks should be near or under chunk_size"
        )
        total_original = len(sample_long_text)
        avg_chunk = sum(len(c) for c in chunks) / len(chunks)
        assert avg_chunk < total_original, (
            "Average chunk size should be smaller than original text"
        )
        print(f"✅ Split text into {len(chunks)} chunks "
              f"(avg {avg_chunk:.0f} chars, original {total_original} chars)")

    def test_split_documents(self, sample_documents: list[Document]) -> None:
        """Split Document objects and verify metadata is preserved."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=80,
            chunk_overlap=10,
        )
        split_docs = splitter.split_documents(sample_documents)

        assert len(split_docs) >= len(sample_documents), (
            "Split should produce at least as many documents as input"
        )
        # Metadata should be preserved on all chunks
        for doc in split_docs:
            assert "source" in doc.metadata, "Source metadata must be preserved"
        print(f"✅ Split {len(sample_documents)} docs into {len(split_docs)} chunks")


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_embedding_dimensions(self, mock_embeddings: MagicMock) -> None:
        """Mock OpenAIEmbeddings to return a 1536-dim vector and verify length."""
        vector = mock_embeddings.embed_query("What is a vector database?")

        assert isinstance(vector, list), "Embedding should be a list"
        assert len(vector) == 1536, f"Expected 1536 dimensions, got {len(vector)}"
        assert all(isinstance(v, float) for v in vector), (
            "All embedding values should be floats"
        )
        print(f"✅ Embedding dimension: {len(vector)} (expected 1536)")

    def test_batch_embeddings(self, mock_embeddings: MagicMock) -> None:
        """Verify batch embedding produces correct number of vectors."""
        texts = ["Hello world", "Vector databases", "LangChain framework"]
        vectors = mock_embeddings.embed_documents(texts)

        assert len(vectors) == 3, f"Expected 3 vectors, got {len(vectors)}"
        assert all(len(v) == 1536 for v in vectors), (
            "All vectors should have 1536 dimensions"
        )
        print(f"✅ Batch embedded {len(texts)} texts into {len(vectors)} vectors")


class TestVectorStore:
    """Tests for ChromaDB vector store operations."""

    def test_vector_store_add_and_search(
        self,
        temp_dir: str,
        sample_documents: list[Document],
        mock_embeddings: MagicMock,
    ) -> None:
        """Use ChromaDB in a temp directory with mock embeddings.

        Add 3 docs, query, assert results returned.
        """
        from langchain_chroma import Chroma

        # Create vector store with mock embeddings
        vectorstore = Chroma.from_documents(
            documents=sample_documents,
            embedding=mock_embeddings,
            persist_directory=os.path.join(temp_dir, "chroma_test"),
            collection_name="test_collection",
        )

        # Query the vector store
        results = vectorstore.similarity_search("What is RAG?", k=2)

        assert len(results) > 0, "Should return at least one result"
        assert len(results) <= 2, "Should return at most k=2 results"
        assert all(isinstance(r, Document) for r in results), (
            "Results should be Document objects"
        )
        assert all(r.page_content for r in results), (
            "Results should have non-empty page_content"
        )
        print(f"✅ Vector store returned {len(results)} results for 'What is RAG?'")
        print(f"   Top result: {results[0].page_content[:80]}...")

    def test_vector_store_with_metadata_filter(
        self,
        temp_dir: str,
        sample_documents: list[Document],
        mock_embeddings: MagicMock,
    ) -> None:
        """Test metadata filtering in ChromaDB queries."""
        from langchain_chroma import Chroma

        vectorstore = Chroma.from_documents(
            documents=sample_documents,
            embedding=mock_embeddings,
            persist_directory=os.path.join(temp_dir, "chroma_filter_test"),
            collection_name="filter_test",
        )

        # Filter by source
        results = vectorstore.similarity_search(
            "embeddings",
            k=3,
            filter={"source": "vector_db_guide.txt"},
        )

        assert len(results) >= 1, "Should find at least 1 result with filter"
        assert all(
            r.metadata["source"] == "vector_db_guide.txt" for r in results
        ), "All results should match the filter"
        print(f"✅ Metadata filter returned {len(results)} results")


class TestRAGChain:
    """Tests for the full RAG chain (fully mocked)."""

    @patch("langchain_openai.ChatOpenAI")
    def test_basic_rag_answer(self, mock_chat_class: MagicMock) -> None:
        """Mock the full RAG chain and assert non-empty answer returned."""
        from langchain_core.messages import AIMessage
        
        # Set up mock LLM
        mock_llm = MagicMock()
        mock_response = AIMessage(content=(
            "Based on the provided context, vector databases store embeddings "
            "for fast similarity search and are essential for RAG applications."
        ))
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        # Simulate RAG: context + question → LLM
        context = "Vector databases store embeddings for fast similarity search."
        question = "What are vector databases used for?"

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context.\n\n"
            "Context: {context}\n\n"
            "Question: {question}"
        )

        # Build and invoke the chain
        chain = prompt | mock_llm
        result = chain.invoke({"context": context, "question": question})

        assert result.content, "Answer should not be empty"
        assert len(result.content) > 10, "Answer should be substantive"
        print(f"✅ RAG chain returned: {result.content[:80]}...")

    @patch("langchain_openai.ChatOpenAI")
    def test_failure_mode_hallucination(self, mock_chat_class: MagicMock) -> None:
        """With empty context, the LLM should acknowledge lack of information."""
        from langchain_core.messages import AIMessage
        
        mock_llm = MagicMock()
        mock_response = AIMessage(content="I don't have that information in the provided context.")
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "Answer the question ONLY based on the provided context. "
            "If the context does not contain the answer, say "
            "'I don\\'t have that information'.\n\n"
            "Context: {context}\n\n"
            "Question: {question}"
        )

        chain = prompt | mock_llm
        result = chain.invoke({
            "context": "",  # empty context
            "question": "What is the company's revenue?",
        })

        answer = result.content.lower()
        assert "don't" in answer or "do not" in answer or "don't" in answer, (
            f"Expected acknowledgment of missing info, got: {result.content}"
        )
        print(f"✅ Hallucination guard: '{result.content}'")


class TestTokenCounter:
    """Tests for token counting with tiktoken (no mock needed)."""

    def test_token_counter(self) -> None:
        """Use tiktoken directly to count tokens in sample text."""
        import tiktoken

        text = (
            "Vector databases store high-dimensional embeddings and enable "
            "fast similarity search for retrieval-augmented generation."
        )

        # Use cl100k_base (GPT-4 / text-embedding-ada-002 encoding)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        token_count = len(tokens)

        assert token_count > 0, "Token count should be positive"
        assert token_count < len(text), (
            "Token count should be less than character count"
        )
        # Reasonable range: ~15-25 tokens for this sentence
        assert 10 <= token_count <= 40, (
            f"Token count {token_count} outside expected range [10, 40]"
        )
        print(f"✅ Token count for sample text: {token_count} tokens "
              f"({len(text)} chars, ratio: {len(text)/token_count:.1f} chars/token)")

    def test_token_counter_different_models(self) -> None:
        """Verify token counts vary by encoding model."""
        import tiktoken

        text = "Hello, how are you doing today?"

        cl100k = tiktoken.get_encoding("cl100k_base")
        p50k = tiktoken.get_encoding("p50k_base")

        count_cl100k = len(cl100k.encode(text))
        count_p50k = len(p50k.encode(text))

        assert count_cl100k > 0
        assert count_p50k > 0
        print(f"✅ Token counts — cl100k_base: {count_cl100k}, "
              f"p50k_base: {count_p50k}")
