"""
07_rag_failure_modes.py — RAG Failure Modes & Diagnostics

Demonstrates the 5 most common failure modes in RAG applications,
shows why they happen, and implements practical fixes for each one.

Failure Modes Covered:
  1. Bad Chunking — naive splitting breaks semantic boundaries
  2. Embedding Mismatch — vocabulary gap between user query and docs
  3. Retrieval Noise — too many irrelevant chunks pollute context
  4. Context Overflow — too many tokens overwhelm LLM attention
  5. Hallucination — LLM fabricates info not grounded in context

Each failure mode includes:
  - A concrete demonstration of the problem
  - A diagnostic explanation of WHY it fails
  - A production fix with before/after comparison
"""

import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Load environment ─────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAGFailureDiagnostics — Central class for all 5 failure modes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RAGFailureDiagnostics:
    """Demonstrates, diagnoses, and fixes common RAG failure modes."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment. "
                "Copy .env.example → .env and add your key."
            )

        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.encoding = tiktoken.encoding_for_model(model_name)

        # Sample text used across multiple failure modes
        self._sample_text = self._load_sample_text()

    # ── Helper ────────────────────────────────────────────────────
    def _load_sample_text(self) -> str:
        """Load company_policy.txt as sample text for demonstrations."""
        policy_file = DOCS_DIR / "company_policy.txt"
        if policy_file.exists():
            return policy_file.read_text(encoding="utf-8")
        # Fallback sample if file is missing
        return textwrap.dedent("""\
            ACME Corporation — Employee Leave & Time-Off Policy.
            Full-time employees are entitled to 15 days of paid annual leave per calendar year.
            Part-time employees receive prorated leave based on their contracted hours.
            Annual leave accrues at a rate of 1.25 days per month.
            Unused leave beyond the carryover limit of 5 days will be forfeited on December 31st.
            Employees must submit leave requests at least 14 days in advance through the HR portal.
            Managers must approve or deny requests within 3 business days.
            All employees are entitled to 10 days of paid sick leave per year.
            Sick leave does not accrue and resets on January 1st each year.
            For absences exceeding 3 consecutive days, employees must provide a medical certificate.
            ACME Corporation provides 16 weeks of paid parental leave for primary caregivers.
            Upon termination, employees will be compensated for any unused accrued annual leave.
            To discontinue service, employees should submit a formal resignation letter.
        """)

    @staticmethod
    def _section_header(number: int, title: str) -> None:
        """Print a formatted section header."""
        print(f"\n{'═' * 70}")
        print(f"  FAILURE MODE {number}: {title}")
        print(f"{'═' * 70}")

    @staticmethod
    def _sub_header(label: str) -> None:
        print(f"\n  ── {label} {'─' * max(1, 55 - len(label))}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE MODE 1 — Bad Chunking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def demonstrate_bad_chunking(self) -> None:
        """Show how naive fixed-size chunking breaks semantic boundaries."""
        self._section_header(1, "BAD CHUNKING")

        text = self._sample_text

        # ── Naive splitting: fixed character boundary ─────────
        self._sub_header("PROBLEM: Naive Fixed-Size Splitting (chunk_size=120)")
        naive_splitter = CharacterTextSplitter(
            separator="",          # Split on nothing — pure character count
            chunk_size=120,
            chunk_overlap=0,
            strip_whitespace=False,
        )
        naive_chunks = naive_splitter.split_text(text)

        broken_count = 0
        for i, chunk in enumerate(naive_chunks[:8]):  # Show first 8
            # Heuristic: chunk is "broken" if it doesn't start after a
            # sentence-ending punctuation or newline
            is_broken = (
                not chunk.strip().endswith((".", "!", "?", "\n"))
                and i < len(naive_chunks) - 1
            )
            status = "❌ BROKEN MID-SENTENCE" if is_broken else "✅ Clean"
            if is_broken:
                broken_count += 1
            preview = chunk.replace("\n", " ").strip()[:80]
            print(f"    Chunk {i:>2}: [{status}] \"{preview}...\"")

        print(f"\n    Summary: {broken_count}/{min(8, len(naive_chunks))} "
              f"chunks shown have broken sentence boundaries.")
        print("    → This destroys semantic coherence. The LLM sees sentence fragments.")

        # ── Fix: RecursiveCharacterTextSplitter ───────────────
        self._sub_header("FIX: RecursiveCharacterTextSplitter (chunk_size=300)")
        smart_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            chunk_size=300,
            chunk_overlap=50,
            strip_whitespace=True,
        )
        smart_chunks = smart_splitter.split_text(text)

        clean_count = 0
        for i, chunk in enumerate(smart_chunks[:6]):  # Show first 6
            ends_clean = chunk.strip().endswith((".", "!", "?", "\n", ":"))
            if ends_clean:
                clean_count += 1
            status = "✅ Clean boundary" if ends_clean else "⚠️  Partial"
            preview = chunk.replace("\n", " ").strip()[:80]
            print(f"    Chunk {i:>2}: [{status}] \"{preview}...\"")

        print(f"\n    Summary: {clean_count}/{min(6, len(smart_chunks))} "
              f"chunks shown have clean semantic boundaries.")
        print("    → RecursiveCharacterTextSplitter tries sentence/paragraph "
              "boundaries first.")
        print("    → Overlap of 50 chars preserves context across chunk edges.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE MODE 2 — Embedding Mismatch (Vocabulary Gap)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def demonstrate_embedding_mismatch(self) -> dict:
        """Show how vocabulary gaps reduce retrieval quality, and fix with query rewriting."""
        self._section_header(2, "EMBEDDING MISMATCH (Vocabulary Gap)")

        query = "How do I cancel my subscription?"
        doc_texts = [
            "Our termination policy requires 30 days notice before discontinuing service.",
            "To discontinue service, employees should submit a formal resignation letter.",
            "The company newsletter is published every Friday at 3 PM.",
            "Lunch menu is available in the break room on the second floor.",
        ]

        # ── Show the vocabulary gap ──────────────────────────
        self._sub_header("PROBLEM: User says 'cancel subscription', docs say 'termination/discontinue'")
        print(f"    Query: \"{query}\"")
        print(f"    Doc 0: \"{doc_texts[0][:70]}...\"")
        print(f"    Doc 1: \"{doc_texts[1][:70]}...\"")

        query_emb = self.embeddings.embed_query(query)
        doc_embs = self.embeddings.embed_documents(doc_texts)

        # Cosine similarity (embeddings are already normalized for OpenAI)
        import numpy as np
        q_vec = np.array(query_emb)
        scores = []
        for i, d_emb in enumerate(doc_embs):
            d_vec = np.array(d_emb)
            cos_sim = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
            scores.append(cos_sim)
            relevance = "✅ RELEVANT" if i < 2 else "❌ IRRELEVANT"
            print(f"    Similarity(query, doc_{i}): {cos_sim:.4f}  [{relevance} doc]")

        gap = scores[0] - scores[2]
        print(f"\n    Gap between best relevant ({scores[0]:.4f}) and best "
              f"irrelevant ({scores[2]:.4f}): {gap:.4f}")
        if gap < 0.1:
            print("    ⚠️  Thin margin! Vocabulary gap makes relevant docs hard to distinguish.")
        else:
            print("    → Embeddings capture SOME semantic similarity despite vocabulary gap.")

        # ── Fix: Query Rewriting ─────────────────────────────
        self._sub_header("FIX: Query Rewriting — generate alternative phrasings")
        alternatives = self.query_rewriting(query)
        print(f"    Original query:      \"{query}\"")
        for j, alt in enumerate(alternatives):
            print(f"    Rewritten query {j+1}:   \"{alt}\"")

        # Show improved retrieval with rewritten queries
        self._sub_header("RESULT: Similarity scores with rewritten queries")
        best_per_doc = list(scores)  # Start with original scores
        for alt_query in alternatives:
            alt_emb = self.embeddings.embed_query(alt_query)
            alt_vec = np.array(alt_emb)
            for i, d_emb in enumerate(doc_embs):
                d_vec = np.array(d_emb)
                alt_sim = float(np.dot(alt_vec, d_vec) / (
                    np.linalg.norm(alt_vec) * np.linalg.norm(d_vec)
                ))
                best_per_doc[i] = max(best_per_doc[i], alt_sim)

        for i, best_score in enumerate(best_per_doc):
            improvement = best_score - scores[i]
            imp_str = f"(+{improvement:.4f})" if improvement > 0.001 else "(no change)"
            print(f"    Doc {i}: original={scores[i]:.4f}  →  "
                  f"best_with_rewrites={best_score:.4f} {imp_str}")

        return {
            "original_query": query,
            "alternatives": alternatives,
            "original_scores": scores,
            "improved_scores": best_per_doc,
        }

    def query_rewriting(self, query: str, n_alternatives: int = 3) -> list[str]:
        """Generate alternative phrasings of a query to bridge vocabulary gaps.

        Args:
            query: The original user query.
            n_alternatives: Number of alternative phrasings to generate.

        Returns:
            List of alternative query strings.
        """
        prompt = (
            f"Generate exactly {n_alternatives} alternative phrasings of this search query. "
            f"Use synonyms, formal/informal variants, and domain-specific terminology. "
            f"Return ONLY the alternative queries, one per line, no numbering.\n\n"
            f"Original query: {query}"
        )
        response = self.llm.invoke(prompt)
        alternatives = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]
        return alternatives[:n_alternatives]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE MODE 3 — Retrieval Noise
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def demonstrate_retrieval_noise(self) -> dict:
        """Show how large k retrieval returns mostly irrelevant chunks."""
        self._section_header(3, "RETRIEVAL NOISE")

        query = "What is the annual leave policy?"

        # Simulate 10 retrieved chunks with varying relevance
        retrieved_chunks = [
            Document(page_content="Full-time employees are entitled to 15 days of paid annual leave per calendar year.", metadata={"source": "policy", "section": "annual_leave"}),
            Document(page_content="Annual leave accrues at a rate of 1.25 days per month and may be carried over up to a maximum of 5 days.", metadata={"source": "policy", "section": "annual_leave"}),
            Document(page_content="The company newsletter is published every Friday at 3 PM.", metadata={"source": "newsletter", "section": "general"}),
            Document(page_content="Lunch menu for the cafeteria changes weekly.", metadata={"source": "cafeteria", "section": "general"}),
            Document(page_content="The parking garage on level B2 is reserved for executives.", metadata={"source": "facilities", "section": "parking"}),
            Document(page_content="Annual company picnic is scheduled for July 15th.", metadata={"source": "events", "section": "social"}),
            Document(page_content="Employees must submit leave requests at least 14 days in advance through the HR portal.", metadata={"source": "policy", "section": "annual_leave"}),
            Document(page_content="The building fire drill is scheduled for next Tuesday.", metadata={"source": "safety", "section": "general"}),
            Document(page_content="WiFi password for the guest network is posted at reception.", metadata={"source": "IT", "section": "general"}),
            Document(page_content="Printer on floor 3 has been moved to room 312.", metadata={"source": "IT", "section": "equipment"}),
        ]

        # Simulate similarity scores (descending, as a real retrieval would return)
        simulated_scores = [0.92, 0.87, 0.71, 0.68, 0.65, 0.63, 0.82, 0.59, 0.55, 0.51]

        # ── Show the noisy retrieval ─────────────────────────
        self._sub_header(f"PROBLEM: k=10 retrieval returns mostly noise")
        print(f"    Query: \"{query}\"")
        print(f"    Retrieved {len(retrieved_chunks)} chunks:\n")

        relevant_count = 0
        for i, (doc, score) in enumerate(zip(retrieved_chunks, simulated_scores)):
            is_relevant = score >= 0.75
            if is_relevant:
                relevant_count += 1
            status = "✅ RELEVANT" if is_relevant else "❌ NOISE"
            preview = doc.page_content[:65]
            print(f"    [{i:>2}] score={score:.2f} {status}  \"{preview}...\"")

        noise_count = len(retrieved_chunks) - relevant_count
        print(f"\n    Result: Only {relevant_count} of {len(retrieved_chunks)} "
              f"chunks are relevant. {noise_count} are pure noise!")
        print("    → Noise dilutes context and confuses the LLM.")

        # ── Fix: Relevance scoring with threshold ────────────
        self._sub_header("FIX: relevance_scorer() with threshold=0.75")
        filtered = self.relevance_scorer(
            query=query,
            documents=retrieved_chunks,
            scores=simulated_scores,
            threshold=0.75,
        )

        print(f"    After filtering (threshold ≥ 0.75):")
        for i, (doc, score) in enumerate(filtered):
            preview = doc.page_content[:70]
            print(f"    [{i}] score={score:.2f} ✅  \"{preview}...\"")

        print(f"\n    Result: {len(retrieved_chunks)} noisy chunks → "
              f"{len(filtered)} relevant chunks")
        print("    → Only high-confidence chunks reach the LLM. Much cleaner context.")

        return {
            "total_retrieved": len(retrieved_chunks),
            "noise_count": noise_count,
            "after_filter": len(filtered),
        }

    def relevance_scorer(
        self,
        query: str,
        documents: list[Document],
        scores: list[float],
        threshold: float = 0.75,
    ) -> list[tuple[Document, float]]:
        """Filter retrieved documents by minimum similarity threshold.

        Args:
            query: The original search query.
            documents: List of retrieved documents.
            scores: Parallel list of similarity scores.
            threshold: Minimum similarity score to keep.

        Returns:
            List of (document, score) tuples that pass the threshold,
            sorted by score descending.
        """
        filtered = [
            (doc, score)
            for doc, score in zip(documents, scores)
            if score >= threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE MODE 4 — Context Overflow
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def demonstrate_context_overflow(self) -> dict:
        """Show how stuffing too many chunks overflows LLM context window."""
        self._section_header(4, "CONTEXT OVERFLOW")

        # Create 10 synthetic document chunks of varying length
        chunks = []
        sections = self._sample_text.split("\n\n")
        for i, section in enumerate(sections):
            if section.strip():
                # Duplicate content to simulate large documents
                content = (section.strip() + "\n") * 3
                chunks.append(Document(
                    page_content=content,
                    metadata={"source": "policy", "chunk_id": i},
                ))
        # Pad to at least 10 chunks
        while len(chunks) < 10:
            chunks.append(Document(
                page_content=f"Additional policy information section {len(chunks)}. " * 40,
                metadata={"source": "policy", "chunk_id": len(chunks)},
            ))

        # ── Show the overflow problem ────────────────────────
        self._sub_header("PROBLEM: Stuffing 10 chunks into context")
        total_tokens = self.token_counter(chunks[:10])
        print(f"    Total documents: {len(chunks[:10])}")
        print(f"    Total tokens:    {total_tokens:,}")
        print(f"    Typical prompt + instructions: ~500 tokens")
        print(f"    Effective context used: {total_tokens + 500:,} tokens")
        print()
        if total_tokens > 3000:
            print("    ⚠️  Over 3,000 tokens of context!")
            print("    → LLM attention degrades on middle chunks ('lost in the middle' effect)")
            print("    → More tokens = more cost, slower response, worse quality")
        else:
            print("    Context is manageable, but in production with larger docs this explodes.")

        per_chunk_tokens = []
        for i, doc in enumerate(chunks[:10]):
            n_tokens = self.token_counter([doc])
            per_chunk_tokens.append(n_tokens)
            bar = "█" * min(50, n_tokens // 10)
            print(f"    Chunk {i:>2}: {n_tokens:>5} tokens  {bar}")

        # ── Fix: Token counting + context trimming ───────────
        self._sub_header("FIX: context_trimmer() — keep most relevant under budget")
        max_tokens = 3000
        trimmed = self.context_trimmer(chunks[:10], max_tokens=max_tokens)
        trimmed_total = self.token_counter(trimmed)

        print(f"    Max token budget:   {max_tokens:,}")
        print(f"    Before trimming:    {len(chunks[:10])} chunks, {total_tokens:,} tokens")
        print(f"    After trimming:     {len(trimmed)} chunks, {trimmed_total:,} tokens")
        print(f"    Tokens saved:       {total_tokens - trimmed_total:,}")

        for i, doc in enumerate(trimmed):
            n_tokens = self.token_counter([doc])
            print(f"    Kept chunk {doc.metadata.get('chunk_id', i):>2}: {n_tokens:>5} tokens")

        # ── Map-reduce pattern for very large context ────────
        self._sub_header("PATTERN: Map-Reduce for very large context")
        print("    When context is too large even after trimming, use map-reduce:")
        print()
        print("    ┌──────────┐   ┌──────────┐   ┌──────────┐")
        print("    │ Chunk 1  │   │ Chunk 2  │   │ Chunk 3  │   ← MAP: summarize each")
        print("    └────┬─────┘   └────┬─────┘   └────┬─────┘")
        print("         │              │              │")
        print("         └──────────────┼──────────────┘")
        print("                        │")
        print("                  ┌─────▼─────┐")
        print("                  │  REDUCE:  │   ← Combine summaries")
        print("                  │  Final    │      into final answer")
        print("                  │  Answer   │")
        print("                  └───────────┘")
        print()
        print("    Implementation: Use LangChain's MapReduceDocumentsChain")
        print("    or manually: summarize each chunk → combine summaries → answer")

        return {
            "total_tokens_before": total_tokens,
            "total_tokens_after": trimmed_total,
            "chunks_before": len(chunks[:10]),
            "chunks_after": len(trimmed),
        }

    def token_counter(self, docs: list[Document]) -> int:
        """Count the total tokens across a list of documents using tiktoken.

        Args:
            docs: List of LangChain Document objects.

        Returns:
            Total token count across all documents.
        """
        total = 0
        for doc in docs:
            total += len(self.encoding.encode(doc.page_content))
        return total

    def context_trimmer(
        self,
        docs: list[Document],
        max_tokens: int = 3000,
    ) -> list[Document]:
        """Trim documents to fit within a token budget, keeping earliest (most relevant) first.

        Assumes documents are pre-sorted by relevance (most relevant first),
        which is the standard output of vector similarity search.

        Args:
            docs: List of Document objects sorted by relevance.
            max_tokens: Maximum total token budget for context.

        Returns:
            Subset of documents that fit within the token budget.
        """
        kept: list[Document] = []
        running_total = 0

        for doc in docs:
            doc_tokens = len(self.encoding.encode(doc.page_content))
            if running_total + doc_tokens <= max_tokens:
                kept.append(doc)
                running_total += doc_tokens
            else:
                # Once we can't fit the next doc, stop
                # (preserves relevance ordering)
                break

        return kept

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE MODE 5 — Hallucination
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def demonstrate_hallucination(self) -> dict:
        """Show LLM hallucination and how to constrain + detect it."""
        self._section_header(5, "HALLUCINATION")

        context = textwrap.dedent("""\
            ACME Corporation provides 16 weeks of paid parental leave for primary
            caregivers and 6 weeks for secondary caregivers. This applies to birth,
            adoption, and foster placement. Parental leave must commence within
            12 months of the qualifying event.
        """).strip()

        question = "What is ACME's parental leave policy, and do they offer childcare benefits?"

        # ── Show the hallucination problem ───────────────────
        self._sub_header("PROBLEM: Unconstrained prompt → Hallucination")

        naive_prompt = (
            f"Answer the following question based on the context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        naive_response = self.llm.invoke(naive_prompt)
        print(f"    Question: \"{question}\"")
        print(f"    Context mentions: parental leave (16 weeks primary, 6 weeks secondary)")
        print(f"    Context does NOT mention: childcare benefits, daycare, subsidies")
        print()
        print(f"    Naive LLM Response:")
        for line in textwrap.wrap(naive_response.content, width=70):
            print(f"      {line}")
        print()
        print("    ⚠️  Check: Does the answer mention childcare/daycare/benefits")
        print("    not in the context? If so, that's a HALLUCINATION.")

        # ── Fix 1: Constrained prompt ────────────────────────
        self._sub_header("FIX 1: constrained_prompt() — strict grounding instructions")
        constrained_answer = self.constrained_prompt(question, context)
        print(f"    Constrained Response:")
        for line in textwrap.wrap(constrained_answer, width=70):
            print(f"      {line}")

        # ── Fix 2: Faithfulness check ────────────────────────
        self._sub_header("FIX 2: faithfulness_check() — score answer against context")

        # Check the naive (potentially hallucinated) answer
        naive_score = self.faithfulness_check(naive_response.content, context)
        print(f"    Naive answer faithfulness:       {naive_score:.2f}/1.00")

        # Check the constrained answer
        constrained_score = self.faithfulness_check(constrained_answer, context)
        print(f"    Constrained answer faithfulness:  {constrained_score:.2f}/1.00")

        print()
        if constrained_score > naive_score:
            print("    ✅ Constrained prompt produces MORE faithful answer.")
        elif constrained_score == naive_score:
            print("    ℹ️  Both answers equally faithful (LLM may not have hallucinated).")
        else:
            print("    ⚠️  Unusual: constrained scored lower. Review prompt template.")

        print()
        print("    Scoring guide:")
        print("      0.90-1.00 → Fully grounded, safe to show user")
        print("      0.70-0.89 → Mostly grounded, minor extrapolations")
        print("      0.50-0.69 → Partially grounded, some unsupported claims")
        print("      0.00-0.49 → Mostly hallucinated, do NOT show to user")

        return {
            "naive_answer": naive_response.content,
            "constrained_answer": constrained_answer,
            "naive_faithfulness": naive_score,
            "constrained_faithfulness": constrained_score,
        }

    def constrained_prompt(self, question: str, context: str) -> str:
        """Generate an answer with strict grounding instructions to prevent hallucination.

        Args:
            question: User's question.
            context: Retrieved context to ground the answer in.

        Returns:
            LLM response constrained to only use information from the context.
        """
        prompt = textwrap.dedent(f"""\
            You are a precise question-answering assistant. Follow these rules STRICTLY:

            1. ONLY use information explicitly stated in the provided context.
            2. If the context does not contain the answer, say "The provided context does
               not contain information about this."
            3. Do NOT add information from your training data or general knowledge.
            4. Do NOT speculate, infer, or extrapolate beyond what is stated.
            5. If the question asks about multiple topics and only some are covered,
               answer what you can and explicitly state what is NOT in the context.
            6. Quote or closely paraphrase the context when possible.

            Context:
            {context}

            Question: {question}

            Answer (grounded ONLY in the context above):
        """)
        response = self.llm.invoke(prompt)
        return response.content

    def faithfulness_check(self, answer: str, context: str) -> float:
        """Score how faithfully an answer reflects the given context.

        Uses the LLM itself as a judge to evaluate whether claims in the
        answer are supported by the context.

        Args:
            answer: The generated answer to evaluate.
            context: The source context the answer should be grounded in.

        Returns:
            Float between 0.0 (completely unfaithful) and 1.0 (fully faithful).
        """
        prompt = textwrap.dedent(f"""\
            You are a faithfulness evaluator. Your job is to determine how faithfully
            an answer reflects ONLY the information in the given context.

            Score from 0.0 to 1.0:
            - 1.0 = Every claim in the answer is directly supported by the context
            - 0.5 = Some claims are supported, some are not in the context
            - 0.0 = The answer is entirely fabricated with no basis in the context

            Context:
            {context}

            Answer to evaluate:
            {answer}

            Respond with ONLY a number between 0.0 and 1.0, nothing else.
        """)
        response = self.llm.invoke(prompt)

        # Parse the score robustly
        try:
            score_text = response.content.strip()
            # Handle cases like "0.85" or "Score: 0.85"
            for token in score_text.split():
                try:
                    score = float(token)
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
            return float(score_text)
        except (ValueError, IndexError):
            print(f"    ⚠️  Could not parse faithfulness score: '{response.content}'")
            return 0.5  # Default to uncertain


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Run all 5 RAG failure mode demonstrations."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        RAG Failure Modes — Diagnostics & Fixes                 ║")
    print("║        Demonstrates 5 common failure patterns in RAG           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        diag = RAGFailureDiagnostics()
    except EnvironmentError as e:
        print(f"\n❌ Setup Error: {e}")
        sys.exit(1)

    # Failure Mode 1: Bad Chunking (no API calls for embedding)
    print("\n\n🔍 Running Failure Mode 1: Bad Chunking...")
    diag.demonstrate_bad_chunking()

    # Failure Mode 2: Embedding Mismatch (requires OpenAI embeddings)
    print("\n\n🔍 Running Failure Mode 2: Embedding Mismatch...")
    try:
        result_2 = diag.demonstrate_embedding_mismatch()
    except Exception as e:
        print(f"    ❌ Skipped: {e}")

    # Failure Mode 3: Retrieval Noise (simulated scores, no API calls)
    print("\n\n🔍 Running Failure Mode 3: Retrieval Noise...")
    result_3 = diag.demonstrate_retrieval_noise()

    # Failure Mode 4: Context Overflow (token counting only)
    print("\n\n🔍 Running Failure Mode 4: Context Overflow...")
    result_4 = diag.demonstrate_context_overflow()

    # Failure Mode 5: Hallucination (requires LLM)
    print("\n\n🔍 Running Failure Mode 5: Hallucination...")
    try:
        result_5 = diag.demonstrate_hallucination()
    except Exception as e:
        print(f"    ❌ Skipped: {e}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  SUMMARY OF RAG FAILURE MODES")
    print(f"{'═' * 70}")
    print("""
    ┌─────┬──────────────────────┬─────────────────────────────────┐
    │  #  │  Failure Mode        │  Fix                            │
    ├─────┼──────────────────────┼─────────────────────────────────┤
    │  1  │  Bad Chunking        │  RecursiveCharacterTextSplitter │
    │  2  │  Embedding Mismatch  │  Query rewriting (3 variants)   │
    │  3  │  Retrieval Noise     │  Similarity threshold filter    │
    │  4  │  Context Overflow    │  Token counting + trimming      │
    │  5  │  Hallucination       │  Constrained prompt + checking  │
    └─────┴──────────────────────┴─────────────────────────────────┘
    """)
    print("✅ All failure mode demonstrations complete.")


if __name__ == "__main__":
    main()
