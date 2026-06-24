"""
Part 6 — GraphRAG Introduction
==============================
Demonstrates using a Knowledge Graph (networkx) combined with RAG to enable
multi-hop reasoning across documents.
"""

import os
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

class GraphRAG:
    """A minimal GraphRAG implementation using NetworkX."""
    
    def __init__(self):
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o"), temperature=0)
        self.graph = nx.DiGraph()
        self.doc_store = {}
        
    def extract_entities(self, doc_id: str, text: str) -> list:
        """Extract (subject, relation, object) triples from text."""
        print(f"  🧠 Extracting entities from {doc_id}...")
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel, Field
        
        class Triples(BaseModel):
            triples: list[list[str]] = Field(description="List of [subject, relation, object] triples")
            
        parser = JsonOutputParser(pydantic_object=Triples)
        
        template = """Extract key entity relationships from the text.
        Format as a JSON object with a 'triples' array containing [subject, relation, object] arrays.
        Example: {{"triples": [["Harrison Chase", "created", "LangChain"]]}}
        
        Text: {text}
        
        {format_instructions}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.llm | parser
        try:
            result = chain.invoke({"text": text})
            return result.get("triples", [])
        except Exception as e:
            print(f"  ⚠️ Extraction failed: {e}")
            return []
            
    def build_graph(self, documents: dict):
        """Build the knowledge graph from documents."""
        print("\n🕸️  BUILDING KNOWLEDGE GRAPH")
        print("="*50)
        
        self.doc_store = documents
        
        for doc_id, text in documents.items():
            triples = self.extract_entities(doc_id, text)
            print(f"     Found {len(triples)} relationships.")
            
            for triple in triples:
                if len(triple) >= 3:
                    sub, rel, obj = triple[0], triple[1], triple[2]
                    # Add edge and store doc_id in edge attributes for provenance
                    self.graph.add_edge(sub, obj, relation=rel, source_doc=doc_id)
                    
        print(f"\n✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
    def extract_query_entities(self, question: str) -> list[str]:
        """Extract main entities from the question."""
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import CommaSeparatedListOutputParser
        
        parser = CommaSeparatedListOutputParser()
        template = "Extract the main entities from this question as a comma-separated list.\nQuestion: {question}\nEntities:"
        
        chain = PromptTemplate.from_template(template) | self.llm | parser
        entities = chain.invoke({"question": question})
        return [e.strip() for e in entities]
        
    def graph_rag_query(self, question: str, max_hops: int = 2) -> str:
        """Answer a question using multi-hop graph traversal."""
        print(f"\n❓ Question: {question}")
        
        # 1. Extract entities from question
        query_entities = self.extract_query_entities(question)
        print(f"  🔍 Query entities: {query_entities}")
        
        # 2. Find starting nodes in graph (fuzzy match)
        start_nodes = []
        for q_ent in query_entities:
            for node in self.graph.nodes():
                if q_ent.lower() in str(node).lower() or str(node).lower() in q_ent.lower():
                    start_nodes.append(node)
                    
        print(f"  📍 Matched graph nodes: {start_nodes}")
        
        if not start_nodes:
            return "Could not map question to knowledge graph."
            
        # 3. Traverse graph (subgraph extraction)
        subgraph_nodes = set(start_nodes)
        for _ in range(max_hops):
            current_layer = list(subgraph_nodes)
            for node in current_layer:
                # Add neighbors
                subgraph_nodes.update(self.graph.predecessors(node))
                subgraph_nodes.update(self.graph.successors(node))
                
        # 4. Extract context facts and docs
        facts = []
        relevant_doc_ids = set()
        
        for u, v, data in self.graph.subgraph(subgraph_nodes).edges(data=True):
            facts.append(f"{u} -[{data.get('relation', 'related_to')}]-> {v}")
            if 'source_doc' in data:
                relevant_doc_ids.add(data['source_doc'])
                
        print(f"  🧠 Extracted {len(facts)} facts across {max_hops} hops.")
        
        # Add original document text for the relevant subgraphs
        doc_context = "\n".join([f"Doc '{doc_id}': {self.doc_store[doc_id]}" for doc_id in relevant_doc_ids])
        
        # 5. Generate answer
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        template = """Answer the question using the provided knowledge graph facts and source documents.
        
        Knowledge Graph Facts:
        {facts}
        
        Source Documents:
        {doc_context}
        
        Question: {question}
        Answer:"""
        
        chain = PromptTemplate.from_template(template) | self.llm | StrOutputParser()
        answer = chain.invoke({
            "facts": "\n".join(facts),
            "doc_context": doc_context,
            "question": question
        })
        
        return answer

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY required.")
        return
        
    print("🕸️  GRAPHRAG DEMO")
    
    # Setup test data designed for multi-hop
    documents = {
        "doc1": "Alice is the CEO of CyberDyne Systems. CyberDyne Systems built Skynet.",
        "doc2": "Skynet is an artificial intelligence network. Skynet initiated judgment day.",
        "doc3": "Bob works for CyberDyne Systems as a programmer. Bob wrote the core algorithm."
    }
    
    grag = GraphRAG()
    grag.build_graph(documents)
    
    # Standard RAG would struggle with this because "Alice" and "judgment day" 
    # are in different documents with no lexical overlap.
    # We need 3 hops: Alice -> CyberDyne -> Skynet -> judgment day
    
    question = "Who is the CEO of the company that built the system that initiated judgment day?"
    
    print("\n" + "-"*50)
    answer = grag.graph_rag_query(question, max_hops=3)
    print("\n🤖 GraphRAG Answer:")
    print(answer)
    print("-"*50)

if __name__ == "__main__":
    main()
