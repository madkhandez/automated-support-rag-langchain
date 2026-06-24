"""
Part 3, File 14: Token Budgeting — Managing Context Window Limits
==================================================================

Demonstrates how to stay within LLM context-window limits when RAG retrieves
more text than the model can process.

Key concepts
------------
* **Token counting** with tiktoken (cl100k_base encoding, used by GPT-4o)
* **Dynamic context window**: fit as many high-relevance chunks as possible
  into a budget = llm_max × safety_margin, reserving room for the prompt
  template and the model's answer.
* **Map-Reduce fallback**: when documents exceed the budget even after
  pruning, summarise each chunk independently (map), then combine
  summaries to answer the question (reduce).
* Detailed token-usage statistics at every step.

Why this matters
----------------
Exceeding the context window silently truncates input or raises errors.
Wasting budget on low-relevance chunks dilutes answer quality.
Token budgeting ensures optimal use of every token.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Constants ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_token_budget"
COLLECTION = "token_budget_demo"

# Prompt templates (their token cost is part of the budget)
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the context does not contain the answer, say "
            "\"I don't have enough information to answer that.\"",
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:",
        ),
    ]
)

MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarise the following document chunk in 2-3 sentences, preserving "
            "key facts, numbers, and proper nouns.",
        ),
        ("human", "{chunk}"),
    ]
)

REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Using the combined summaries below, "
            "answer the user's question thoroughly.",
        ),
        (
            "human",
            "Summaries:\n{summaries}\n\n"
            "Question: {question}\n\n"
            "Answer:",
        ),
    ]
)

TEST_QUESTIONS = [
    "Summarize all types of employee leave and their durations.",
    "Explain the different vector database indexing strategies and their trade-offs.",
    "What are the company's policies on remote work and termination?",
]


# ═══════════════════════════════════════════════════════════════════════
# TokenBudgetManager
# ═══════════════════════════════════════════════════════════════════════
class TokenBudgetManager:
    """Manages LLM context-window budgets using tiktoken for accurate
    token counting."""

    def __init__(
        self,
        llm_max_tokens: int = 8000,
        safety_margin: float = 0.8,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.llm_max_tokens = llm_max_tokens
        self.safety_margin = safety_margin
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

        # Derived budgets
        self.total_budget = int(llm_max_tokens * safety_margin)
        self.prompt_reserve = self._count_prompt_template_tokens()
        self.context_budget = self.total_budget - self.prompt_reserve

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=1024)
        self.vectorstore: Chroma | None = None

    # ── Token counting ───────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in *text* using the tiktoken encoder."""
        return len(self.encoding.encode(text))

    def count_doc_tokens(self, doc: Document) -> int:
        """Count tokens in a Document's page_content."""
        return self.count_tokens(doc.page_content)

    def _count_prompt_template_tokens(self) -> int:
        """Estimate the fixed token cost of the QA prompt template
        (with placeholders removed)."""
        # Render with empty placeholders to measure template overhead
        rendered = QA_PROMPT.format(context="", question="")
        return self.count_tokens(rendered) + 50  # +50 buffer for safety

    # ── Build index ──────────────────────────────────────────────────
    def load_and_index(self, docs_dir: Path) -> list[Document]:
        """Load and index documents, return the chunks."""
        print("\n📂  Loading documents …")
        raw_docs: list[Document] = []
        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix in {".txt", ".md"}:
                loader = TextLoader(str(fpath), encoding="utf-8")
                raw_docs.extend(loader.load())
                print(f"   ✓ Loaded {fpath.name}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"   📄 Chunks: {len(chunks)}")

        # Print per-chunk token stats
        token_counts = [self.count_doc_tokens(c) for c in chunks]
        print(f"   🔢 Token stats per chunk:")
        print(f"      Min: {min(token_counts)}, Max: {max(token_counts)}, "
              f"Avg: {sum(token_counts)/len(token_counts):.0f}, "
              f"Total: {sum(token_counts)}")

        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        print(f"   ✓ ChromaDB index ready\n")
        return chunks

    # ── Dynamic context window ───────────────────────────────────────
    def dynamic_context_window(
        self,
        docs: list[Document],
        question: str = "",
    ) -> dict[str, Any]:
        """Select as many documents as fit within the token budget.

        Documents are assumed to be in relevance order (highest first).
        Returns a dict with included/excluded docs and token stats.
        """
        question_tokens = self.count_tokens(question) if question else 0
        available = self.context_budget - question_tokens

        included: list[Document] = []
        excluded: list[Document] = []
        tokens_used = 0

        print(f"  🎯  Token Budget Allocation:")
        print(f"     LLM max tokens:      {self.llm_max_tokens:>6}")
        print(f"     Safety margin:        {self.safety_margin:>6.0%}")
        print(f"     Total budget:         {self.total_budget:>6}")
        print(f"     Prompt template:     -{self.prompt_reserve:>6}")
        print(f"     Question tokens:     -{question_tokens:>6}")
        print(f"     Available for context:{available:>6}")
        print()

        for i, doc in enumerate(docs):
            doc_tokens = self.count_doc_tokens(doc)
            if tokens_used + doc_tokens <= available:
                included.append(doc)
                tokens_used += doc_tokens
                status = "✅ INCLUDED"
            else:
                excluded.append(doc)
                status = "❌ EXCLUDED (over budget)"
            print(
                f"     Chunk {i+1:>2}: {doc_tokens:>4} tokens  "
                f"[running: {tokens_used:>5}/{available}]  {status}"
            )

        budget_pct = (tokens_used / available * 100) if available > 0 else 0
        print(f"\n  📊  Context Budget Usage:")
        print(f"     Tokens used:    {tokens_used:>5} / {available}  ({budget_pct:.1f}%)")
        print(f"     Chunks included: {len(included)}")
        print(f"     Chunks excluded: {len(excluded)}")

        return {
            "included": included,
            "excluded": excluded,
            "tokens_used": tokens_used,
            "budget": available,
            "budget_pct": budget_pct,
        }

    # ── Standard QA chain ────────────────────────────────────────────
    def answer_with_budget(
        self, question: str, docs: list[Document]
    ) -> dict[str, Any]:
        """Answer a question using only the docs that fit in the budget."""
        budget_result = self.dynamic_context_window(docs, question)
        included = budget_result["included"]

        if not included:
            return {
                "answer": "No documents fit within the token budget.",
                "method": "none",
                **budget_result,
            }

        context = "\n\n---\n\n".join(doc.page_content for doc in included)
        chain = QA_PROMPT | self.llm | StrOutputParser()

        t0 = time.perf_counter()
        answer = chain.invoke({"context": context, "question": question})
        latency = (time.perf_counter() - t0) * 1000

        answer_tokens = self.count_tokens(answer)
        print(f"\n  ✍️  Answer generated ({answer_tokens} tokens, {latency:.0f} ms)")

        return {
            "answer": answer,
            "method": "direct",
            "answer_tokens": answer_tokens,
            "latency_ms": latency,
            **budget_result,
        }

    # ── Map-Reduce chain ─────────────────────────────────────────────
    def map_reduce_chain(
        self, question: str, docs: list[Document]
    ) -> dict[str, Any]:
        """Summarise each doc independently (map), then combine summaries
        and answer the question (reduce).

        Used when the full document set exceeds the context budget.
        """
        total_tokens = sum(self.count_doc_tokens(d) for d in docs)
        print(f"\n  🗺️  Map-Reduce Pipeline")
        print(f"     Input: {len(docs)} chunks, {total_tokens} tokens total")
        print(f"     Budget: {self.context_budget} tokens")
        print(f"     Strategy: Summarise each chunk, then combine\n")

        # ── MAP phase ────────────────────────────────────────────────
        map_chain = MAP_PROMPT | self.llm | StrOutputParser()
        summaries: list[str] = []
        map_tokens_in = 0
        map_tokens_out = 0

        t0 = time.perf_counter()
        for i, doc in enumerate(docs, 1):
            chunk_tokens = self.count_doc_tokens(doc)
            summary = map_chain.invoke({"chunk": doc.page_content})
            summary_tokens = self.count_tokens(summary)
            map_tokens_in += chunk_tokens
            map_tokens_out += summary_tokens
            summaries.append(summary)
            compression = (1 - summary_tokens / chunk_tokens) * 100 if chunk_tokens > 0 else 0
            print(
                f"     MAP chunk {i:>2}: {chunk_tokens:>4} → {summary_tokens:>3} tokens  "
                f"({compression:.0f}% compression)"
            )
        map_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  📊  MAP phase: {map_tokens_in} → {map_tokens_out} tokens "
              f"({(1 - map_tokens_out/map_tokens_in)*100:.0f}% total compression, {map_ms:.0f} ms)")

        # ── REDUCE phase ─────────────────────────────────────────────
        combined_summaries = "\n\n".join(
            f"[Summary {i}]: {s}" for i, s in enumerate(summaries, 1)
        )
        combined_tokens = self.count_tokens(combined_summaries)
        print(f"\n  📋  REDUCE input: {combined_tokens} tokens "
              f"(fits budget: {'✅' if combined_tokens <= self.context_budget else '⚠️ still large'})")

        reduce_chain = REDUCE_PROMPT | self.llm | StrOutputParser()
        t0 = time.perf_counter()
        answer = reduce_chain.invoke(
            {"summaries": combined_summaries, "question": question}
        )
        reduce_ms = (time.perf_counter() - t0) * 1000
        answer_tokens = self.count_tokens(answer)

        print(f"  ✍️  Final answer: {answer_tokens} tokens ({reduce_ms:.0f} ms)")

        return {
            "answer": answer,
            "method": "map_reduce",
            "map_tokens_in": map_tokens_in,
            "map_tokens_out": map_tokens_out,
            "combined_summary_tokens": combined_tokens,
            "answer_tokens": answer_tokens,
            "map_ms": map_ms,
            "reduce_ms": reduce_ms,
            "total_ms": map_ms + reduce_ms,
            "num_chunks": len(docs),
        }

    # ── Smart answer: choose strategy automatically ──────────────────
    def smart_answer(
        self, question: str, k: int = 10
    ) -> dict[str, Any]:
        """Retrieve documents and automatically choose between direct
        context injection and map-reduce based on token budget."""
        assert self.vectorstore is not None

        docs = self.vectorstore.similarity_search(question, k=k)
        total_tokens = sum(self.count_doc_tokens(d) for d in docs)
        question_tokens = self.count_tokens(question)
        available = self.context_budget - question_tokens

        print(f"\n  🤖  Smart Strategy Selection:")
        print(f"     Retrieved {len(docs)} docs, {total_tokens} total tokens")
        print(f"     Context budget: {available} tokens")

        if total_tokens <= available:
            print(f"     → Direct injection (all docs fit in budget)\n")
            return self.answer_with_budget(question, docs)
        else:
            print(f"     → Map-Reduce required (docs exceed budget by "
                  f"{total_tokens - available} tokens)\n")
            # First try budget-fit with pruning
            budget_result = self.dynamic_context_window(docs, question)
            if len(budget_result["excluded"]) > 2:
                # Many docs excluded — use map-reduce to capture all info
                print(f"\n  ⚠️  {len(budget_result['excluded'])} chunks excluded — "
                      f"falling back to map-reduce for completeness\n")
                return self.map_reduce_chain(question, docs)
            else:
                # Few docs excluded — budget pruning is fine
                return self.answer_with_budget(question, docs)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the token budgeting demonstration."""
    print("=" * 72)
    print("  Part 3 · File 14 — Token Budgeting & Map-Reduce")
    print("=" * 72)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    manager = TokenBudgetManager(llm_max_tokens=8000, safety_margin=0.8)

    # Show budget breakdown
    print(f"\n  📐  Token Budget Configuration:")
    print(f"     Encoding:        {manager.encoding_name}")
    print(f"     LLM max:         {manager.llm_max_tokens} tokens")
    print(f"     Safety margin:   {manager.safety_margin:.0%}")
    print(f"     Total budget:    {manager.total_budget} tokens")
    print(f"     Prompt reserve:  {manager.prompt_reserve} tokens")
    print(f"     Context budget:  {manager.context_budget} tokens")

    chunks = manager.load_and_index(DOCS_DIR)

    # ── Demo 1: Dynamic context window ───────────────────────────────
    print(f"\n{'═' * 72}")
    print("  DEMO 1: Dynamic Context Window")
    print(f"{'═' * 72}")

    question = TEST_QUESTIONS[0]
    print(f"\n  Question: \"{question}\"")
    docs = manager.vectorstore.similarity_search(question, k=10)  # type: ignore
    result = manager.answer_with_budget(question, docs)
    print(f"\n  📝  Answer ({result['method']}):")
    print(f"  {result['answer'][:300]}{'…' if len(result['answer']) > 300 else ''}")

    # ── Demo 2: Map-Reduce for large doc sets ────────────────────────
    print(f"\n{'═' * 72}")
    print("  DEMO 2: Map-Reduce Chain")
    print(f"{'═' * 72}")

    question2 = TEST_QUESTIONS[1]
    print(f"\n  Question: \"{question2}\"")

    # Use all chunks to force map-reduce
    print(f"  (Using ALL {len(chunks)} chunks to demonstrate map-reduce)")
    result2 = manager.map_reduce_chain(question2, chunks)
    print(f"\n  📝  Answer ({result2['method']}):")
    print(f"  {result2['answer'][:300]}{'…' if len(result2['answer']) > 300 else ''}")

    # ── Demo 3: Smart strategy selection ─────────────────────────────
    print(f"\n{'═' * 72}")
    print("  DEMO 3: Smart Strategy Selection")
    print(f"{'═' * 72}")

    for question in TEST_QUESTIONS:
        print(f"\n  Question: \"{question}\"")
        result3 = manager.smart_answer(question, k=10)
        print(f"\n  📝  Answer ({result3['method']}):")
        answer_preview = result3["answer"][:200]
        print(f"  {answer_preview}{'…' if len(result3['answer']) > 200 else ''}")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  📊  TOKEN BUDGETING SUMMARY")
    print(f"{'═' * 72}")
    print(
        "\n  Key takeaways:"
        "\n  1. Always count tokens with tiktoken BEFORE sending to the LLM"
        "\n  2. Reserve 20% safety margin for prompt overhead + answer generation"
        "\n  3. Include highest-relevance chunks first, drop lowest-relevance ones"
        "\n  4. Fall back to map-reduce when too many docs are excluded"
        "\n  5. Map-reduce trades latency for completeness — use when recall matters"
        "\n"
    )

    # Cleanup
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🧹  Cleaned up temporary ChromaDB directory.\n")


if __name__ == "__main__":
    main()
