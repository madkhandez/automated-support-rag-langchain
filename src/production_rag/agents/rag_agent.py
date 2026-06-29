import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .agent_state import AgentState
from ..core.vector_store import VectorStoreFactory
from ..core.llm_factory import LLMFactory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class RagAgent:
    """LangGraph agent with self-correction loop."""
    
    def __init__(self):
        self.vector_store = VectorStoreFactory().get_vector_store()
        self.llm, self.active_provider = LLMFactory.get_llm()
        logger.info(f"RagAgent using LLM provider: {self.active_provider}")
        self.checkpointer = MemorySaver()
        self.app = self._build_graph()
        
    def _build_graph(self):
        """Construct the state graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("transform_query", self._transform_query_node)
        workflow.add_node("grade_generation", self._grade_generation_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge after grading docs
        workflow.add_conditional_edges(
            "grade_documents",
            self._check_docs_relevance,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", "grade_generation")
        
        # Conditional edge after grading generation
        workflow.add_conditional_edges(
            "grade_generation",
            self._check_hallucination,
            {
                "end": END,
                "retry": "generate",
                "max_retries": END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
        
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve documents."""
        question = state["question"]
        print(f"---RETRIEVE: {question}---")
        
        # Default fallback if vector store isn't fully initialized in tests
        docs = []
        if hasattr(self.vector_store, 'similarity_search'):
            docs = self.vector_store.similarity_search(question, k=4)
        
        return {"documents": docs, "question": question}
        
    def _grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """Filter out irrelevant documents."""
        print("---GRADE DOCUMENTS---")
        docs = state["documents"]
        question = state["question"]
        
        # In a full implementation, we'd use LLM to grade each doc
        # For simplicity, we assume they are relevant if retrieved
        filtered_docs = docs
        needs_web_search = len(filtered_docs) == 0
        
        return {"documents": filtered_docs, "needs_web_search": needs_web_search}
        
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer."""
        print("---GENERATE---")
        question = state["question"]
        docs = state["documents"]
        count = state.get("generation_count", 0)
        
        context = "\n\n".join(d.page_content for d in docs) if docs else "No context found."
        
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"context": context, "question": question})
        except Exception:
            answer = "I apologize, but I encountered an error generating the answer."
            
        return {"answer": answer, "generation_count": count + 1}
        
    def _transform_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite the query to improve retrieval."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        
        template = """You are optimizing a search query. 
        Look at the original query and formulate an improved, more specific version.
        Original: {question}
        Improved query:"""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            better_question = chain.invoke({"question": question})
        except Exception:
            better_question = question
            
        return {"question": better_question}
        
    def _grade_generation_node(self, state: AgentState) -> Dict[str, Any]:
        """Check if answer is grounded in context."""
        print("---GRADE GENERATION---")
        # In full implementation, LLM checks if answer hallucinates
        # For simplicity, we assume true
        return {"is_grounded": True}
        
    # --- Edge Conditions ---
    def _check_docs_relevance(self, state: AgentState) -> str:
        """Determine next step after grading documents."""
        if state.get("needs_web_search", False):
            return "transform_query"
        return "generate"
        
    def _check_hallucination(self, state: AgentState) -> str:
        """Determine next step after generation."""
        count = state.get("generation_count", 0)
        
        if count >= 3:
            return "max_retries"
            
        if state.get("is_grounded", True):
            return "end"
            
        return "retry"
        
    def run_agent(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Run the agent and return result."""
        config = {"configurable": {"thread_id": session_id}}
        
        # Initialize state
        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "generation_count": 0,
            "is_grounded": False,
            "needs_web_search": False
        }
        
        # Run graph
        result = self.app.invoke(initial_state, config=config)
        
        # Extract sources from documents
        sources = []
        seen_filenames = set()
        if "documents" in result:
            for doc in result["documents"]:
                if hasattr(doc, "metadata"):
                    filename = doc.metadata.get("source", "Unknown")
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        excerpt = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
                        sources.append({"filename": filename, "excerpt": excerpt})
            
        return {
            "answer": result.get("answer", "No answer generated."),
            "sources": sources
        }
