import os
from typing import Any, Dict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import logging
from production_rag.core.llm_factory import LLMFactory
from production_rag.core.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)

class ProductionRAGPipeline:
    """Core RAG pipeline for processing queries and ingesting documents."""
    
    def __init__(self) -> None:
        self.vs_factory = VectorStoreFactory()
        
        # Initialize components
        self.llm, self.active_provider = LLMFactory.get_llm()
        logger.info(f"RAG pipeline using LLM provider: {self.active_provider}")
        self.vector_store = self.vs_factory.get_vector_store()
        
        # Setup RAG prompt (retriever is created per-query with session filter)
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    def _build_chain(self, session_id: str | None = None):
        """Build a RAG chain with an optional session-scoped retriever.

        Args:
            session_id: If provided, the retriever only returns documents
                        whose metadata ``session_id`` matches this value.
        """
        search_kwargs: dict = {"k": 4}
        if session_id:
            search_kwargs["filter"] = {"session_id": session_id}

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def query(self, question: str, session_id: str | None = None) -> Dict[str, Any]:
        """Process a question through the RAG pipeline.

        Args:
            question: The user's natural-language question.
            session_id: Scope retrieval to this session's documents only.
        """
        rag_chain = self._build_chain(session_id=session_id)
        response = rag_chain.invoke({"input": question})
        
        answer = response["answer"]
        source_docs = response.get("context", [])
        
        sources = []
        seen_filenames = set()
        for doc in source_docs:
            filename = doc.metadata.get("source", "Unknown")
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                excerpt = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
                sources.append({"filename": filename, "excerpt": excerpt})
                
        # Rough token count estimation for demonstration
        token_count = len(answer) // 4
                
        return {
            "answer": answer,
            "sources": sources,
            "token_count": token_count,
        }

    def ingest_documents(self, file_path: str, original_filename: str = None, session_id: str | None = None) -> Dict[str, Any]:
        """Ingest a file into the vector store. Returns chunk count.

        Args:
            file_path: Path to the file to ingest.
            original_filename: Display name for the source metadata.
            session_id: Tag every chunk with this session for isolation.
        """
        # Use original filename if provided, else extract from path
        filename = original_filename if original_filename else os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            # Fallback to TextLoader for txt/md
            loader = TextLoader(file_path, encoding="utf-8")
            
        docs = loader.load()
        
        # Update metadata to original filename + session scoping
        for doc in docs:
            doc.metadata["source"] = filename
            if session_id:
                doc.metadata["session_id"] = session_id
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(docs)
        if chunks:
            self.vs_factory.add_documents(chunks)
            
        return {"chunks_indexed": len(chunks)}
