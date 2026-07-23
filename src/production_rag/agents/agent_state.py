"""
Part 5 — Agent State: Shared state schema for the LangGraph RAG Agent.

Defines the ``AgentState`` TypedDict that flows through every node in the
self-correcting RAG agent graph.  Each field is documented so that node
implementations have a clear contract.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes.

    Attributes:
        question:          The user's original (or rewritten) question.
        documents:         Retrieved context documents.
        answer:            The generated answer text.
        generation_count:  How many generate attempts have been made (for retry cap).
        is_grounded:       Whether the latest answer is grounded in context.
        needs_web_search:  Flag set by the grader when retrieval quality is poor.
        session_id:        Session identifier for document isolation filtering.
    """

    question: str
    documents: list[Document]
    answer: str
    generation_count: int
    is_grounded: bool
    needs_web_search: bool
    session_id: str


def create_initial_state(question: str) -> AgentState:
    """Helper to build a clean initial state for a new query.

    Args:
        question: The user's natural-language question.

    Returns:
        A fully initialised ``AgentState`` dict.
    """
    return AgentState(
        question=question,
        documents=[],
        answer="",
        generation_count=0,
        is_grounded=False,
        needs_web_search=False,
    )


# ── Standalone entrypoint ────────────────────────────────────────────
def main() -> None:
    """Quick demo of the AgentState dataclass."""
    print("=" * 60)
    print("Part 5 · Agent State Demo")
    print("=" * 60)

    state = create_initial_state("What is the company leave policy?")

    print(f"\n  question:          {state['question']}")
    print(f"  documents:         {state['documents']}")
    print(f"  answer:            {state['answer']!r}")
    print(f"  generation_count:  {state['generation_count']}")
    print(f"  is_grounded:       {state['is_grounded']}")
    print(f"  needs_web_search:  {state['needs_web_search']}")

    print("\n✅ AgentState demo complete.")


if __name__ == "__main__":
    main()
