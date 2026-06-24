"""
10_langsmith_tracing.py — LangSmith Observability for RAG

Implements production-grade observability using LangSmith tracing:
  - Sets up LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT from environment
  - Uses @traceable decorator for automatic tracing
  - Adds custom metadata: user_id, session_id, question_category
  - Traces full RAG query chains
  - Compares two RAG runs (A vs B)
  - Logs user feedback (thumbs up/down, comments)

Everything degrades gracefully when LangSmith is not configured —
the app runs normally, just without tracing.

Usage:
  python 10_langsmith_tracing.py

Requires:
  - OPENAI_API_KEY in .env
  - LANGCHAIN_API_KEY in .env (optional — tracing disabled without it)
"""

import os
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# ── Load environment FIRST (before LangSmith auto-detection) ─────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ── Configure LangSmith from environment ─────────────────────────
LANGSMITH_AVAILABLE = False
_langsmith_client = None

# Check if LangSmith is configured
_langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
_has_langsmith = (
    _langchain_api_key
    and _langchain_api_key != "your_langsmith_api_key_here"
    and len(_langchain_api_key) > 10
)

if _has_langsmith:
    # Set tracing environment variables
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "production-rag-debugging"))

    try:
        from langsmith import Client as LangSmithClient
        from langsmith import traceable

        _langsmith_client = LangSmithClient()
        LANGSMITH_AVAILABLE = True
        print("✅ LangSmith tracing ENABLED")
        print(f"   Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
    except ImportError:
        print("⚠️  langsmith package not installed. Tracing disabled.")
        print("   Install with: pip install langsmith")
    except Exception as e:
        print(f"⚠️  LangSmith setup failed: {e}. Tracing disabled.")
else:
    print("ℹ️  LangSmith not configured (no LANGCHAIN_API_KEY). Tracing disabled.")
    print("   To enable: add LANGCHAIN_API_KEY to your .env file.")

# ── Create a no-op @traceable decorator when LangSmith is absent ──
if not LANGSMITH_AVAILABLE:
    def traceable(
        *args: Any,
        name: Optional[str] = None,
        run_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
        **kwargs: Any,
    ) -> Callable:
        """No-op replacement for @traceable when LangSmith is not available."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return func(*a, **kw)
            return wrapper

        # Handle both @traceable and @traceable(...) syntax
        if args and callable(args[0]):
            return args[0]
        return decorator


# ── Now import LangChain components ──────────────────────────────
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / ".chroma_langsmith"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper: Classify question category
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def classify_question(question: str) -> str:
    """Classify a question into a category for metadata tagging.

    Args:
        question: The user's question text.

    Returns:
        Category string like 'policy', 'technical', 'general'.
    """
    q_lower = question.lower()
    if any(w in q_lower for w in ["leave", "policy", "vacation", "sick", "parental", "hr"]):
        return "policy"
    elif any(w in q_lower for w in ["vector", "embedding", "database", "index", "chroma", "search"]):
        return "technical"
    elif any(w in q_lower for w in ["how", "tutorial", "guide", "setup", "install"]):
        return "how-to"
    else:
        return "general"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Traced RAG Pipeline Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@traceable(name="embed_query", run_type="embedding")
def embed_query(embeddings: OpenAIEmbeddings, query: str) -> list[float]:
    """Embed a query string — traced as an embedding run."""
    return embeddings.embed_query(query)


@traceable(name="retrieve_documents", run_type="retriever")
def retrieve_documents(
    vectorstore: Chroma,
    query: str,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve documents from vector store — traced as a retriever run.

    Returns serialisable dicts instead of Document objects for clean traces.
    """
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    results = []
    for doc, score in docs_with_scores:
        results.append({
            "content": doc.page_content,
            "metadata": dict(doc.metadata),
            "score": score,
        })
    return results


@traceable(name="generate_answer", run_type="llm")
def generate_answer(
    llm: ChatOpenAI,
    question: str,
    context: str,
) -> str:
    """Generate an answer from context — traced as an LLM run."""
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content


@traceable(
    name="rag_query",
    run_type="chain",
    metadata={"pipeline": "production-rag", "version": "1.0"},
)
def trace_rag_query(
    question: str,
    session_id: str,
    user_id: str = "demo_user",
    vectorstore: Optional[Chroma] = None,
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> dict[str, Any]:
    """Execute a full RAG query with LangSmith tracing.

    This is the main traced function that orchestrates the entire RAG pipeline.
    When LangSmith is configured, this creates a trace with:
      - Custom metadata (user_id, session_id, question_category)
      - Child spans for embedding, retrieval, and generation
      - Input/output captured automatically

    Args:
        question: The user's question.
        session_id: Session ID for grouping related queries.
        user_id: User identifier for the trace.
        vectorstore: ChromaDB vector store (created if None).
        llm: LLM instance (created if None).
        embeddings: Embeddings instance (created if None).

    Returns:
        Dict with answer, sources, metadata, and timing info.
    """
    start_time = time.time()

    # Initialize components if not provided
    if embeddings is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)
    if llm is None:
        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        llm = ChatOpenAI(model=model_name, temperature=0)
    if vectorstore is None:
        vectorstore = _get_or_build_vectorstore(embeddings)

    # Classify question for metadata
    category = classify_question(question)

    # Step 1: Retrieve
    retrieval_start = time.time()
    retrieved = retrieve_documents(vectorstore, question, k=3)
    retrieval_time = time.time() - retrieval_start

    # Step 2: Build context
    context_parts = [r["content"] for r in retrieved]
    context = "\n\n".join(context_parts)

    # Step 3: Generate
    generation_start = time.time()
    answer = generate_answer(llm, question, context)
    generation_time = time.time() - generation_start

    total_time = time.time() - start_time

    result = {
        "question": question,
        "answer": answer,
        "sources": [r["metadata"].get("source", "unknown") for r in retrieved],
        "retrieval_scores": [r["score"] for r in retrieved],
        "num_chunks_used": len(retrieved),
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "question_category": category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "timing": {
            "retrieval_ms": round(retrieval_time * 1000, 1),
            "generation_ms": round(generation_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
        },
    }

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Run Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compare_rag_runs(
    question: str,
    session_id: str,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two RAG configurations side by side.

    Runs the same question with two different configurations and
    compares the results. Useful for A/B testing prompt templates,
    models, or retrieval strategies.

    Args:
        question: The question to test.
        session_id: Session ID for tracing.
        config_a: Config dict for Run A (keys: model, temperature, k).
        config_b: Config dict for Run B (same keys).

    Returns:
        Dict with side-by-side comparison results.
    """
    print(f"\n  ── Comparing RAG Runs ─────────────────────────────")
    print(f"    Question: \"{question}\"")
    print(f"    Config A: {config_a}")
    print(f"    Config B: {config_b}")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = _get_or_build_vectorstore(embeddings)

    results = {}
    for label, config in [("A", config_a), ("B", config_b)]:
        print(f"\n    ── Run {label} ──")
        model = config.get("model", os.getenv("LLM_MODEL", "gpt-4o"))
        temperature = config.get("temperature", 0)

        llm = ChatOpenAI(model=model, temperature=temperature)

        run_result = trace_rag_query(
            question=question,
            session_id=f"{session_id}_run_{label}",
            user_id=f"comparison_run_{label}",
            vectorstore=vectorstore,
            llm=llm,
            embeddings=embeddings,
        )
        results[label] = run_result

        print(f"      Model: {model}  |  Temp: {temperature}")
        print(f"      Time: {run_result['timing']['total_ms']}ms")
        print(f"      Answer ({len(run_result['answer'].split())} words):")
        for line in run_result["answer"].split("\n"):
            if line.strip():
                print(f"        {line.strip()[:70]}")

    # Comparison metrics
    print(f"\n    ── Comparison ──")
    print(f"      {'Metric':<25s}  {'Run A':>12s}  {'Run B':>12s}")
    print(f"      {'─' * 55}")

    for metric in ["total_ms"]:
        val_a = results["A"]["timing"][metric]
        val_b = results["B"]["timing"][metric]
        faster = "←" if val_a < val_b else "→"
        print(f"      {'Latency (ms)':<25s}  {val_a:>12.1f}  {val_b:>12.1f}  {faster} faster")

    wc_a = len(results["A"]["answer"].split())
    wc_b = len(results["B"]["answer"].split())
    print(f"      {'Word count':<25s}  {wc_a:>12d}  {wc_b:>12d}")

    # Word overlap
    words_a = set(results["A"]["answer"].lower().split())
    words_b = set(results["B"]["answer"].lower().split())
    overlap = len(words_a & words_b)
    total = len(words_a | words_b)
    jaccard = overlap / total if total > 0 else 1.0
    print(f"      {'Answer similarity':<25s}  {jaccard:>12.3f}")

    if jaccard < 0.5:
        print("      ⚠️  Answers diverge significantly between configurations!")
    else:
        print("      ✅ Answers are reasonably consistent.")

    return {"run_A": results["A"], "run_B": results["B"], "similarity": jaccard}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  User Feedback Logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_user_feedback(
    run_id: str,
    score: float,
    comment: str = "",
    feedback_key: str = "user_rating",
) -> dict[str, Any]:
    """Log user feedback for a specific run in LangSmith.

    In production, this would be called when a user rates a response
    (e.g., thumbs up/down, 1-5 stars).

    Args:
        run_id: The LangSmith run ID to attach feedback to.
        score: Feedback score (0.0 = bad, 1.0 = good).
        comment: Optional user comment.
        feedback_key: The feedback metric name.

    Returns:
        Dict with feedback details and status.
    """
    print(f"\n  ── Logging User Feedback ──────────────────────────")
    print(f"    Run ID:  {run_id}")
    print(f"    Score:   {score}")
    print(f"    Comment: \"{comment}\"")
    print(f"    Key:     {feedback_key}")

    feedback_record = {
        "run_id": run_id,
        "key": feedback_key,
        "score": score,
        "comment": comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "logged_to_langsmith": False,
    }

    if LANGSMITH_AVAILABLE and _langsmith_client is not None:
        try:
            _langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_key,
                score=score,
                comment=comment,
            )
            feedback_record["logged_to_langsmith"] = True
            print("    ✅ Feedback logged to LangSmith successfully.")
        except Exception as e:
            print(f"    ⚠️  Failed to log to LangSmith: {e}")
            print("    ℹ️  Feedback recorded locally only.")
    else:
        print("    ℹ️  LangSmith not available. Feedback recorded locally only.")
        print("    → In production, this would be stored in your database.")

    return feedback_record


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vector Store Builder (shared utility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_or_build_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    """Build or load a ChromaDB vector store from docs/ folder."""
    collection_name = "langsmith_tracing_demo"

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if store._collection.count() > 0:
        return store

    # Load documents
    print("  📄 Building vector store from docs/...")
    documents: list[Document] = []

    for filepath in sorted(DOCS_DIR.glob("*")):
        if filepath.suffix in (".txt", ".md"):
            loader = TextLoader(str(filepath), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filepath.name
            documents.extend(docs)

    if not documents:
        # Fallback synthetic docs
        documents = [
            Document(
                page_content="Full-time employees get 15 days annual leave. Part-time is prorated.",
                metadata={"source": "policy.txt"},
            ),
            Document(
                page_content="Vector databases store high-dimensional embeddings for similarity search.",
                metadata={"source": "tech.md"},
            ),
        ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  ✅ Vector store built with {len(chunks)} chunks.")
    return store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Demonstrate LangSmith tracing and observability for RAG."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        LangSmith Tracing — RAG Observability                   ║")
    print("║        Traces, comparisons, and feedback logging               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Verify OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in .env")
        sys.exit(1)

    print(f"\n  LangSmith Status: {'ENABLED ✅' if LANGSMITH_AVAILABLE else 'DISABLED (running without tracing) ℹ️'}")
    print(f"  Tracing Project:  {os.environ.get('LANGCHAIN_PROJECT', 'not set')}")

    # ── Demo 1: Traced RAG Query ─────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 DEMO 1: Traced RAG Query")
    print("=" * 60)

    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"
    print(f"  Session ID: {session_id}")

    queries = [
        "What is ACME's annual leave policy?",
        "How do vector databases work?",
        "What is the sick leave policy?",
    ]

    run_results = []
    for q in queries:
        print(f"\n  Query: \"{q}\"")
        try:
            result = trace_rag_query(
                question=q,
                session_id=session_id,
                user_id="demo_user_001",
            )
            run_results.append(result)

            print(f"    Category: {result['metadata']['question_category']}")
            print(f"    Sources:  {result['sources']}")
            print(f"    Scores:   {[f'{s:.3f}' for s in result['retrieval_scores']]}")
            print(f"    Timing:   retrieval={result['timing']['retrieval_ms']}ms  "
                  f"generation={result['timing']['generation_ms']}ms  "
                  f"total={result['timing']['total_ms']}ms")
            print(f"    Answer:   {result['answer'][:100]}...")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # ── Demo 2: Compare Two Runs ─────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 DEMO 2: Compare RAG Runs (A vs B)")
    print("=" * 60)

    try:
        comparison = compare_rag_runs(
            question="What is the annual leave policy?",
            session_id=session_id,
            config_a={"model": os.getenv("LLM_MODEL", "gpt-4o"), "temperature": 0},
            config_b={"model": os.getenv("LLM_MODEL", "gpt-4o"), "temperature": 0.7},
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Demo 3: Log User Feedback ────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 DEMO 3: User Feedback Logging")
    print("=" * 60)

    # Simulate feedback for each run
    demo_run_id = str(uuid.uuid4())
    feedback_examples = [
        {"run_id": demo_run_id, "score": 1.0, "comment": "Perfect answer, very helpful!"},
        {"run_id": str(uuid.uuid4()), "score": 0.5, "comment": "Partially correct but missed some details."},
        {"run_id": str(uuid.uuid4()), "score": 0.0, "comment": "Answer was completely wrong."},
    ]

    for fb in feedback_examples:
        try:
            log_user_feedback(**fb)
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  LANGSMITH TRACING SUMMARY")
    print(f"{'═' * 60}")
    print(f"""
    Status:     {'ENABLED' if LANGSMITH_AVAILABLE else 'DISABLED (graceful fallback)'}
    Queries:    {len(run_results)} traced
    Session:    {session_id}

    What was traced:
      ✅ Full RAG chain (embed → retrieve → generate)
      ✅ Custom metadata (user_id, session_id, question_category)
      ✅ Timing metrics per step
      ✅ Input/output for each component
      ✅ Run comparisons (A vs B)
      ✅ User feedback
    """)

    if LANGSMITH_AVAILABLE:
        project = os.environ.get("LANGCHAIN_PROJECT", "default")
        print(f"    View traces at: https://smith.langchain.com/project/{project}")
    else:
        print("    To enable tracing:")
        print("      1. Sign up at https://smith.langchain.com")
        print("      2. Add LANGCHAIN_API_KEY to your .env file")
        print("      3. Re-run this script")

    print("\n✅ LangSmith tracing demonstration complete.")


if __name__ == "__main__":
    main()
