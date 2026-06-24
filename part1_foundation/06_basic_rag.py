"""
Part 1 — Basic RAG System
=========================
A complete RAG pipeline: Load -> Split -> Embed -> Store -> Retrieve -> Generate.
Uses LangChain Expression Language (LCEL).
"""

import os
from dotenv import load_dotenv

load_dotenv()

class BasicRAGSystem:
    """A complete basic RAG implementation."""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "basic_rag"):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            temperature=0
        )
        self._Chroma = Chroma
        self.vectorstore = None
        self.chain = None

    def build_index(self, docs_dir: str):
        """1-4: Load, Split, Embed, Store."""
        from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print(f"\n📥 1. Loading documents from {docs_dir}...")
        
        # Loaders for different types
        loaders = {
            ".txt": DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader),
            ".pdf": DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        }
        
        raw_docs = []
        for ext, loader in loaders.items():
            try:
                docs = loader.load()
                raw_docs.extend(docs)
                print(f"  ✅ Loaded {len(docs)} {ext} files")
            except Exception as e:
                print(f"  ⚠️  Could not load {ext} files: {e}")

        if not raw_docs:
            print("❌ No documents loaded. Check the docs_dir path.")
            return False

        print(f"\n✂️  2. Splitting {len(raw_docs)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"  ✅ Created {len(chunks)} chunks")

        print("\n🧠 3-4. Embedding and Storing in ChromaDB...")
        self.vectorstore = self._Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )
        print(f"  ✅ Vector store created at {self.persist_dir}")
        return True

    def _setup_chain(self):
        """5-7: Retrieve, Prompt, Generate using LCEL."""
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        if not self.vectorstore:
            self.vectorstore = self._Chroma(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )

        # 5. Retrieve
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # Format retrieved docs into a single string
        def format_docs(docs):
            return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)

        # 6. Prompt
        template = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say 'I don't have that information.'
Always cite your sources.

Context: {context}
Question: {question}
Answer:"""
        prompt = PromptTemplate.from_template(template)

        # 7. Chain (LCEL)
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Keep retriever accessible for debugging/citations
        self.retriever = retriever

    def query(self, question: str) -> str:
        """Run the RAG chain for a single question."""
        if not self.chain:
            self._setup_chain()

        print(f"\n❓ Question: {question}")
        print("⏳ Generating answer...")
        
        # For demonstration, we'll invoke the retriever separately to show sources
        docs = self.retriever.invoke(question)
        print("\n📚 Retrieved Sources:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            # Extract just filename for cleaner output
            filename = os.path.basename(source)
            print(f"  [{i}] {filename} (similarity search)")

        answer = self.chain.invoke(question)
        print(f"\n🤖 Answer:\n{answer}")
        return answer

    def interactive_chat(self):
        """Run an interactive chat loop."""
        print("\n" + "="*50)
        print("💬 Basic RAG Chat (type 'exit' to quit)")
        print("="*50)
        
        while True:
            question = input("\nUser: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            if not question.strip():
                continue
                
            self.query(question)

def main():
    """Run the Basic RAG System."""
    print("🚀 BASIC RAG SYSTEM")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in .env to run this module.")
        return

    # Use the docs folder from the project root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    docs_dir = os.path.join(base_dir, "docs")
    persist_dir = os.path.join(base_dir, ".basic_rag_db")

    rag = BasicRAGSystem(persist_dir=persist_dir)
    
    # Check if we need to build index
    if not os.path.exists(persist_dir):
        success = rag.build_index(docs_dir)
        if not success:
            return
    else:
        print(f"\n📂 Using existing vector store at {persist_dir}")

    # Test query
    rag.query("What is the company policy on annual leave?")
    rag.query("How do vector embeddings work?")
    
    # Optional interactive chat
    # rag.interactive_chat()

    print("\n✅ Basic RAG System demo complete!")

if __name__ == "__main__":
    main()
