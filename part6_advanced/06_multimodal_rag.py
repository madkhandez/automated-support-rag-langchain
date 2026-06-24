"""
Part 6 — Multimodal RAG
=======================
Demonstrates processing PDFs as images, extracting structured descriptions
of charts/tables using Vision LLMs, and performing RAG over those descriptions.
"""

import os
import io
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class MultimodalRAG:
    """RAG system capable of processing visual information (charts, tables)."""
    
    def __init__(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        # We need a vision-capable model
        self.vision_llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)
        self.text_llm = ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o"), temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self._Chroma = Chroma
        self.vector_store = None
        
    def pdf_to_images(self, pdf_path: str) -> List[Any]:
        """Convert PDF pages to images. Requires PyMuPDF (fitz) or pdf2image."""
        print(f"\n📄 Converting PDF to images: {pdf_path}")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
                # Convert to bytes
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
            print(f"  ✅ Extracted {len(images)} page images.")
            return images
        except ImportError:
            print("  ⚠️ PyMuPDF (fitz) not installed. Using mock images for demo.")
            return [b"mock_image_bytes_1", b"mock_image_bytes_2"]
            
    def vision_describe_page(self, img_bytes: bytes, page_num: int) -> str:
        """Use Vision LLM to describe the page, especially tables and charts."""
        print(f"  👁️  Analyzing page {page_num} with Vision LLM...")
        from langchain_core.messages import HumanMessage
        import base64
        
        # If using mock data
        if img_bytes.startswith(b"mock_image"):
            if page_num == 1:
                return "Page 1 contains a title 'Q3 Financial Results'. There is a bar chart showing revenue growth from Q1 ($1.2M) to Q2 ($1.5M) to Q3 ($2.1M)."
            else:
                return "Page 2 contains a table showing expenses. Marketing: $500k, R&D: $800k, Operations: $300k. Total expenses: $1.6M."

        # Real vision API call
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        
        prompt = """Describe the contents of this page in detail.
        Pay special attention to any charts, graphs, or tables.
        For tables: extract the rows and columns clearly.
        For charts: describe the axes, trends, and specific data points visible.
        For text: summarize the main points."""
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        )
        
        try:
            response = self.vision_llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"  ⚠️ Vision API failed: {e}")
            return "Failed to analyze image."
            
    def index_pdf(self, pdf_path: str):
        """Process a PDF using the multimodal pipeline and index it."""
        from langchain_core.documents import Document
        
        images = self.pdf_to_images(pdf_path)
        documents = []
        
        for i, img_bytes in enumerate(images, 1):
            description = self.vision_describe_page(img_bytes, i)
            
            # Create a document where the content is the LLM's rich description
            doc = Document(
                page_content=description,
                metadata={
                    "source": pdf_path,
                    "page": i,
                    "type": "visual_description"
                }
            )
            documents.append(doc)
            
        print("\n📥 Indexing visual descriptions into vector store...")
        self.vector_store = self._Chroma.from_documents(documents, self.embeddings)
        print("  ✅ Indexing complete.")
        
    def visual_rag_query(self, question: str) -> str:
        """Answer a question using the visual descriptions."""
        if not self.vector_store:
            return "Error: No documents indexed."
            
        print(f"\n❓ Question: {question}")
        
        # Retrieve relevant visual descriptions
        docs = self.vector_store.similarity_search(question, k=2)
        
        print("\n📚 Retrieved Visual Context:")
        for doc in docs:
            print(f"  - Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
            
        # Generate answer
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        context = "\n\n".join([f"Page {d.metadata.get('page')}: {d.page_content}" for d in docs])
        
        template = """You are a helpful assistant analyzing documents.
        Based on the following descriptions of document pages (which may include descriptions of charts and tables), answer the user's question.
        
        Document Descriptions:
        {context}
        
        Question: {question}
        Answer:"""
        
        chain = PromptTemplate.from_template(template) | self.text_llm | StrOutputParser()
        
        print("\n🤖 Generating answer...")
        answer = chain.invoke({"context": context, "question": question})
        return answer

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY required.")
        return
        
    print("📊 MULTIMODAL RAG DEMO")
    print("="*70)
    
    rag = MultimodalRAG()
    
    # We'll use a mock path since we fallback to mock data if PyMuPDF isn't installed
    pdf_path = "financial_report.pdf"
    
    # Standard RAG using PyPDFLoader would garble the table on page 2
    # and completely miss the data in the bar chart on page 1.
    
    # 1. Process PDF visually
    rag.index_pdf(pdf_path)
    
    # 2. Ask questions requiring visual data understanding
    print("\n" + "-"*50)
    q1 = "What was the revenue growth from Q1 to Q3?"
    ans1 = rag.visual_rag_query(q1)
    print(f"\nAnswer: {ans1}")
    
    print("\n" + "-"*50)
    q2 = "What were the R&D expenses?"
    ans2 = rag.visual_rag_query(q2)
    print(f"\nAnswer: {ans2}")

if __name__ == "__main__":
    main()
