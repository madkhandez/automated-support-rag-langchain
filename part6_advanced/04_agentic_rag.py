"""
Part 6 — Agentic RAG (Self-Correcting Retrieval)
================================================
Demonstrates a LangGraph-based RAG agent that can evaluate its own answers,
rewrite queries, and fall back to web search when local retrieval fails.
"""

import os
from typing import Dict, Any, List, TypedDict
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    """State for the self-correcting RAG agent."""
    question: str
    documents: List[Any]
    answer: str
    generation_count: int
    is_grounded: bool
    quality_score: int
    web_fallback: bool

class AgenticRAG:
    """A self-correcting RAG system using LangGraph."""
    
    def __init__(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langgraph.graph import StateGraph, END
        from langchain_core.documents import Document
        
        self.llm = ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o"), temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # Setup a simple vector store for demo
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        
        # Create a tiny in-memory DB for the demo
        docs = [
            Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "tech_docs"}),
            Document(page_content="Vector databases store embeddings for similarity search.", metadata={"source": "tech_docs"}),
            Document(page_content="Employees get 15 days of annual leave.", metadata={"source": "policy"})
        ]
        self.vector_store = Chroma.from_documents(docs, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        
        self.app = self._build_graph()

    def _build_graph(self):
        """Construct the self-correcting agent graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("rewrite", self._rewrite_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        
        # Conditional edge after retrieval
        workflow.add_conditional_edges(
            "retrieve",
            self._check_retrieval,
            {
                "generate": "generate",
                "web_search": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "judge")
        
        # Conditional edge after judging
        workflow.add_conditional_edges(
            "judge",
            self._check_quality,
            {
                "end": END,
                "rewrite": "rewrite",
                "max_retries": END
            }
        )
        
        workflow.add_edge("rewrite", "retrieve")
        
        return workflow.compile()
        
    # --- Nodes ---
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve documents from local vector store."""
        print(f"  🔍 [RETRIEVE] Searching for: '{state['question']}'")
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs}
        
    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Fallback web search when local docs aren't sufficient."""
        print(f"  🌐 [WEB SEARCH] Local retrieval insufficient. Searching web for: '{state['question']}'")
        from langchain_core.documents import Document
        
        # Mock web search (replace with Tavily in real app)
        mock_result = f"Web result for {state['question']}: This is mock data from the internet."
        if "einstein" in state["question"].lower():
            mock_result = "Albert Einstein was a German-born theoretical physicist."
            
        docs = state["documents"] + [Document(page_content=mock_result, metadata={"source": "web"})]
        return {"documents": docs, "web_fallback": True}
        
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer from documents."""
        print("  🧠 [GENERATE] Creating answer from context...")
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        docs = state["documents"]
        context = "\n".join([d.page_content for d in docs])
        
        template = """Answer based ONLY on context. If not in context, say you don't know.
        Context: {context}
        Question: {question}
        Answer:"""
        
        chain = PromptTemplate.from_template(template) | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": state["question"]})
        
        count = state.get("generation_count", 0) + 1
        return {"answer": answer, "generation_count": count}
        
    def _judge_node(self, state: AgentState) -> Dict[str, Any]:
        """Score answer quality 1-10."""
        print("  ⚖️  [JUDGE] Evaluating answer quality...")
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        template = """Score this answer 1-10 based on how well it answers the question.
        If it says 'I don't know', score it 1.
        Just output the number.
        Question: {question}
        Answer: {answer}
        """
        
        chain = PromptTemplate.from_template(template) | self.llm | StrOutputParser()
        try:
            score_str = chain.invoke({"question": state["question"], "answer": state["answer"]})
            # Clean up the output to just get the number
            score_str = ''.join(filter(str.isdigit, score_str))
            score = int(score_str) if score_str else 5
        except:
            score = 5
            
        print(f"     Score: {score}/10")
        return {"quality_score": score}
        
    def _rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite query for better retrieval."""
        print("  📝 [REWRITE] Answer quality too low. Rewriting query...")
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        template = """Rewrite this query to be more specific for a search engine.
        Query: {question}
        Rewritten query:"""
        
        chain = PromptTemplate.from_template(template) | self.llm | StrOutputParser()
        new_q = chain.invoke({"question": state["question"]})
        print(f"     New query: '{new_q}'")
        
        return {"question": new_q}
        
    # --- Edges ---
    def _check_retrieval(self, state: AgentState) -> str:
        """Route to web search if local docs are poor."""
        # Simple heuristic: if we retrieved less than 1 doc, or we specifically asked something not in docs
        if not state["documents"] or "einstein" in state["question"].lower():
            return "web_search"
        return "generate"
        
    def _check_quality(self, state: AgentState) -> str:
        """Route based on quality score."""
        if state.get("generation_count", 0) >= 3:
            print("  🛑 [END] Max retries reached.")
            return "max_retries"
            
        if state.get("quality_score", 0) >= 7:
            print("  ✅ [END] Answer is good enough.")
            return "end"
            
        return "rewrite"
        
    def test_query(self, query: str):
        """Run a test query through the graph."""
        print(f"\n" + "="*70)
        print(f"TESTING QUERY: {query}")
        print("="*70)
        
        initial_state = {
            "question": query,
            "documents": [],
            "answer": "",
            "generation_count": 0,
            "quality_score": 0,
            "web_fallback": False
        }
        
        result = self.app.invoke(initial_state)
        
        print(f"\nFinal Answer: {result['answer']}")
        print(f"Iterations: {result['generation_count']}")
        print(f"Web Fallback Used: {result.get('web_fallback', False)}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY required.")
        return
        
    print("🤖 AGENTIC RAG DEMO")
    agent = AgenticRAG()
    
    # Query 1: Local retrieval works perfectly (1 iteration)
    agent.test_query("How many days of leave do employees get?")
    
    # Query 2: Fails local, falls back to web search
    agent.test_query("Who is Albert Einstein?")
    
    # Query 3: Vague query that requires rewriting (multiple iterations)
    # The initial doc retrieval won't have enough context, score will be low, it will rewrite
    agent.test_query("What is the framework?")

if __name__ == "__main__":
    main()
