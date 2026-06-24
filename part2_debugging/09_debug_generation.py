"""
09_debug_generation.py — Generation Debugging Toolkit

Provides the GenerationDebugger class with 5 diagnostic tools for
understanding and fixing generation quality issues in RAG pipelines.

Tools:
  1. test_prompt_variations()  — compare multiple prompt templates
  2. test_temperature_impact() — generate at different temperatures
  3. detect_hallucination()    — check if claims exist in context
  4. measure_answer_quality()  — LLM-as-judge scoring (relevance, completeness, faithfulness)
  5. test_context_ordering()   — reorder docs and check if answer changes

Usage:
  python 09_debug_generation.py

Requires:
  - OPENAI_API_KEY in .env
"""

import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ── Load environment ─────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Prompt Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT_TEMPLATES: dict[str, str] = {
    "basic": (
        "Answer the question based on the context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "structured": (
        "You are a helpful assistant. Use ONLY the provided context to answer.\n"
        "If the answer isn't in the context, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a clear, structured answer:"
    ),
    "strict_grounded": (
        "You are a precise question-answering system. Follow these rules:\n"
        "1. ONLY use information from the context below.\n"
        "2. If the context doesn't contain the answer, respond with "
        "'Information not found in provided context.'\n"
        "3. Do NOT use any outside knowledge.\n"
        "4. Cite specific parts of the context when possible.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (grounded only in context):"
    ),
    "chain_of_thought": (
        "You are an analytical assistant. Answer step by step.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Think through this step by step:\n"
        "1. What relevant information is in the context?\n"
        "2. How does it relate to the question?\n"
        "3. What is the answer based on the context?\n\n"
        "Final Answer:"
    ),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GenerationDebugger
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GenerationDebugger:
    """Diagnostic tools for debugging generation quality in RAG pipelines.

    Uses LLM-as-judge patterns to evaluate answer quality, detect
    hallucinations, and measure sensitivity to prompt design and
    context ordering.
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment. "
                "Copy .env.example → .env and add your key."
            )

        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        """Join document contents into a single context string."""
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"chunk_{i}")
            parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _wrap_print(text: str, indent: int = 6, width: int = 68) -> None:
        """Print text with word-wrapping and indentation."""
        for line in textwrap.wrap(text, width=width):
            print(" " * indent + line)

    @staticmethod
    def _parse_float(text: str, default: float = 0.5) -> float:
        """Robustly extract a float from LLM output."""
        # Try to find a decimal number in the text
        matches = re.findall(r"\d+\.?\d*", text)
        for m in matches:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return val
            elif 1 < val <= 10:
                return val / 10.0  # LLM gave score on 1-10 scale
        return default

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 1: Test Prompt Variations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def test_prompt_variations(
        self,
        question: str,
        context: str,
        templates: Optional[dict[str, str]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Test multiple prompt templates and compare output quality.

        Args:
            question: The user question.
            context: The retrieved context string.
            templates: Dict of {name: template_string}. Uses defaults if None.
                       Templates should contain {context} and {question} placeholders.

        Returns:
            Dict mapping template_name → {answer, word_count, has_disclaimer}.
        """
        if templates is None:
            templates = PROMPT_TEMPLATES

        print(f"\n  ── Testing Prompt Variations ──────────────────────")
        print(f"    Question: \"{question}\"")
        print(f"    Context length: {len(context)} chars")
        print(f"    Templates: {list(templates.keys())}")

        results: dict[str, dict[str, Any]] = {}

        for name, template in templates.items():
            print(f"\n    ── Template: '{name}' {'─' * max(1, 40 - len(name))}")

            prompt = template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # Analysis metrics
            word_count = len(answer.split())
            has_disclaimer = any(
                phrase in answer.lower()
                for phrase in [
                    "not found", "not mentioned", "doesn't contain",
                    "not in the context", "no information",
                    "information not found",
                ]
            )
            sentence_count = len(re.split(r'[.!?]+', answer.strip()))

            results[name] = {
                "answer": answer,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "has_disclaimer": has_disclaimer,
            }

            print(f"      Words: {word_count}  |  Sentences: {sentence_count}  |  "
                  f"Disclaimer: {'Yes' if has_disclaimer else 'No'}")
            print(f"      Answer:")
            self._wrap_print(answer, indent=8, width=62)

        # Comparison summary
        print(f"\n    ── Comparison Summary ──")
        print(f"      {'Template':<20s}  {'Words':>6s}  {'Sentences':>10s}  {'Disclaimer'}")
        print(f"      {'─' * 55}")
        for name, info in results.items():
            disc = "✅ Yes" if info["has_disclaimer"] else "❌ No"
            print(f"      {name:<20s}  {info['word_count']:>6d}  "
                  f"{info['sentence_count']:>10d}  {disc}")

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 2: Test Temperature Impact
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def test_temperature_impact(
        self,
        question: str,
        context: str,
        temperatures: Optional[list[float]] = None,
    ) -> dict[float, dict[str, Any]]:
        """Generate answers at different temperatures and compare.

        Lower temperature → more deterministic, factual
        Higher temperature → more creative, varied (riskier for RAG)

        Args:
            question: The user question.
            context: The retrieved context string.
            temperatures: List of temperature values to test.

        Returns:
            Dict mapping temperature → {answer, word_count, unique_words}.
        """
        if temperatures is None:
            temperatures = [0.0, 0.3, 0.7, 1.0]

        print(f"\n  ── Testing Temperature Impact ─────────────────────")
        print(f"    Question: \"{question}\"")
        print(f"    Temperatures: {temperatures}")

        prompt = PROMPT_TEMPLATES["structured"].format(
            context=context, question=question
        )

        results: dict[float, dict[str, Any]] = {}

        for temp in temperatures:
            print(f"\n    ── Temperature: {temp} {'─' * 45}")

            temp_llm = ChatOpenAI(model=self.model_name, temperature=temp)
            response = temp_llm.invoke(prompt)
            answer = response.content.strip()

            words = answer.lower().split()
            word_count = len(words)
            unique_words = len(set(words))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0

            results[temp] = {
                "answer": answer,
                "word_count": word_count,
                "unique_words": unique_words,
                "lexical_diversity": lexical_diversity,
            }

            print(f"      Words: {word_count}  |  Unique: {unique_words}  |  "
                  f"Lexical diversity: {lexical_diversity:.2f}")
            print(f"      Answer:")
            self._wrap_print(answer, indent=8, width=62)

        # Consistency analysis
        print(f"\n    ── Analysis ──")
        print(f"      {'Temp':>5s}  {'Words':>6s}  {'Unique':>7s}  {'Diversity':>10s}")
        print(f"      {'─' * 35}")
        for temp, info in results.items():
            print(f"      {temp:>5.1f}  {info['word_count']:>6d}  "
                  f"{info['unique_words']:>7d}  {info['lexical_diversity']:>10.3f}")

        # Check if answers diverge significantly at high temp
        if len(results) >= 2:
            low_temp_answer = results[min(temperatures)]["answer"]
            high_temp_answer = results[max(temperatures)]["answer"]
            low_words = set(low_temp_answer.lower().split())
            high_words = set(high_temp_answer.lower().split())
            overlap = len(low_words & high_words)
            total = len(low_words | high_words)
            jaccard = overlap / total if total > 0 else 1.0
            print(f"\n      Word overlap (temp {min(temperatures)} vs {max(temperatures)}): "
                  f"{jaccard:.2f} Jaccard similarity")
            if jaccard < 0.5:
                print("      ⚠️  High divergence! Temperature significantly changes output.")
                print("      → For RAG, use temperature 0.0-0.3 for factual consistency.")
            else:
                print("      ✅ Answers are relatively consistent across temperatures.")

        print("\n      Recommendation: Use temperature=0 for factual RAG queries.")
        print("      Use temperature=0.3 for slightly varied but still grounded answers.")

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 3: Detect Hallucination
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def detect_hallucination(
        self,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """Check if claims in the answer exist in the context.

        Uses the LLM as a judge to decompose the answer into individual
        claims, then verifies each claim against the context.

        Args:
            answer: The generated answer to check.
            context: The source context.

        Returns:
            Dict with claim-level analysis and overall hallucination score.
        """
        print(f"\n  ── Detecting Hallucination ────────────────────────")

        # Step 1: Extract claims from the answer
        claim_prompt = textwrap.dedent(f"""\
            Extract all factual claims from the following answer. List each claim
            on a separate line. Include only factual statements, not opinions or
            qualifiers. Be thorough — every piece of information should be a claim.

            Answer: {answer}

            Claims (one per line):
        """)
        claim_response = self.llm.invoke(claim_prompt)
        claims = [
            line.strip().lstrip("- ").lstrip("0123456789.)")
            for line in claim_response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        print(f"    Extracted {len(claims)} claims from the answer:")
        for i, claim in enumerate(claims):
            print(f"      {i+1}. {claim[:70]}")

        # Step 2: Verify each claim against context
        print(f"\n    Verifying each claim against context...")
        verified_claims: list[dict[str, Any]] = []

        for claim in claims:
            verify_prompt = textwrap.dedent(f"""\
                Does the following context SUPPORT this specific claim?
                Answer with exactly one of: SUPPORTED, NOT_SUPPORTED, PARTIAL

                Context:
                {context}

                Claim: {claim}

                Verdict (SUPPORTED / NOT_SUPPORTED / PARTIAL):
            """)
            verdict_response = self.llm.invoke(verify_prompt)
            verdict_text = verdict_response.content.strip().upper()

            if "NOT_SUPPORTED" in verdict_text:
                verdict = "NOT_SUPPORTED"
                icon = "❌"
            elif "PARTIAL" in verdict_text:
                verdict = "PARTIAL"
                icon = "⚠️"
            else:
                verdict = "SUPPORTED"
                icon = "✅"

            claim_result = {
                "claim": claim,
                "verdict": verdict,
            }
            verified_claims.append(claim_result)
            print(f"      {icon} [{verdict:<14s}] {claim[:60]}")

        # Step 3: Calculate hallucination score
        total = len(verified_claims)
        supported = sum(1 for c in verified_claims if c["verdict"] == "SUPPORTED")
        partial = sum(1 for c in verified_claims if c["verdict"] == "PARTIAL")
        unsupported = sum(1 for c in verified_claims if c["verdict"] == "NOT_SUPPORTED")

        # Score: 1.0 = no hallucination, 0.0 = fully hallucinated
        if total > 0:
            faithfulness_score = (supported + 0.5 * partial) / total
        else:
            faithfulness_score = 1.0

        result = {
            "claims": verified_claims,
            "total_claims": total,
            "supported": supported,
            "partial": partial,
            "unsupported": unsupported,
            "faithfulness_score": faithfulness_score,
        }

        print(f"\n    Results:")
        print(f"      Total claims:      {total}")
        print(f"      Supported:         {supported} ✅")
        print(f"      Partially supported: {partial} ⚠️")
        print(f"      Not supported:     {unsupported} ❌")
        print(f"      Faithfulness score: {faithfulness_score:.2f}/1.00")

        if unsupported > 0:
            print(f"\n    🚨 HALLUCINATION DETECTED: {unsupported} unsupported claim(s)!")
            print("    → Use stricter prompt templates (see test_prompt_variations)")
            print("    → Add faithfulness check as a post-processing gate")
        else:
            print(f"\n    ✅ No hallucination detected. All claims are grounded.")

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 4: Measure Answer Quality
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def measure_answer_quality(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict[str, float]:
        """Use LLM-as-judge to score answer quality on three dimensions.

        Dimensions:
          - Relevance: Does the answer address the question?
          - Completeness: Does the answer cover all aspects from the context?
          - Faithfulness: Is every claim grounded in the context?

        Args:
            question: The original question.
            answer: The generated answer.
            context: The retrieved context.

        Returns:
            Dict with scores for each dimension and an overall score.
        """
        print(f"\n  ── Measuring Answer Quality (LLM-as-Judge) ───────")
        print(f"    Question: \"{question[:60]}\"")
        print(f"    Answer length: {len(answer.split())} words")

        dimensions = {
            "relevance": (
                "Rate how well the answer addresses the specific question asked. "
                "Score 1.0 if the answer directly and completely addresses the question. "
                "Score 0.5 if it partially addresses the question. "
                "Score 0.0 if it does not address the question at all."
            ),
            "completeness": (
                "Rate how completely the answer covers all relevant information from the context. "
                "Score 1.0 if all relevant information from the context is included. "
                "Score 0.5 if some relevant information is missing. "
                "Score 0.0 if most relevant information is missing."
            ),
            "faithfulness": (
                "Rate how faithfully the answer reflects ONLY information in the context. "
                "Score 1.0 if every statement is supported by the context. "
                "Score 0.5 if some statements go beyond the context. "
                "Score 0.0 if the answer is mostly fabricated."
            ),
        }

        scores: dict[str, float] = {}

        for dimension, criteria in dimensions.items():
            judge_prompt = textwrap.dedent(f"""\
                You are an impartial quality evaluator. Evaluate the answer on this dimension:

                Dimension: {dimension.upper()}
                Criteria: {criteria}

                Context:
                {context}

                Question: {question}

                Answer: {answer}

                Respond with ONLY a number between 0.0 and 1.0:
            """)

            response = self.llm.invoke(judge_prompt)
            score = self._parse_float(response.content, default=0.5)
            scores[dimension] = score

            # Visual bar
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            rating = "Excellent" if score >= 0.9 else "Good" if score >= 0.7 else "Fair" if score >= 0.5 else "Poor"
            print(f"    {dimension:<15s}: {score:.2f}  [{bar}]  {rating}")

        # Overall composite score (weighted average)
        weights = {"relevance": 0.35, "completeness": 0.30, "faithfulness": 0.35}
        overall = sum(scores[dim] * weights[dim] for dim in scores)
        scores["overall"] = overall

        bar_len = int(overall * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {'OVERALL':<15s}: {overall:.2f}  [{bar}]")

        # Grade
        if overall >= 0.85:
            print("    Grade: A — Production ready ✅")
        elif overall >= 0.70:
            print("    Grade: B — Good, minor improvements possible")
        elif overall >= 0.55:
            print("    Grade: C — Acceptable but needs improvement ⚠️")
        else:
            print("    Grade: D — Poor quality, needs significant work ❌")

        return scores

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Tool 5: Test Context Ordering
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def test_context_ordering(
        self,
        question: str,
        docs: list[Document],
    ) -> dict[str, dict[str, Any]]:
        """Reorder context documents and check if the answer changes.

        Tests the "lost in the middle" effect: LLMs tend to focus more on
        the beginning and end of context, potentially missing information
        in the middle.

        Args:
            question: The user question.
            docs: List of Document objects (at least 3 recommended).

        Returns:
            Dict mapping ordering_name → {answer, word_count, context_order}.
        """
        print(f"\n  ── Testing Context Ordering Sensitivity ───────────")
        print(f"    Question: \"{question}\"")
        print(f"    Documents: {len(docs)}")

        if len(docs) < 2:
            print("    ⚠️  Need at least 2 documents to test ordering effects.")
            return {}

        orderings: dict[str, list[Document]] = {
            "original": docs,
            "reversed": list(reversed(docs)),
            "most_relevant_first_last": (
                [docs[0]] + docs[2:-1] + [docs[-1]] + [docs[1]]
                if len(docs) >= 3
                else docs
            ),
        }

        # If enough docs, add "most relevant in the middle" ordering
        if len(docs) >= 4:
            middle_idx = len(docs) // 2
            middle_order = (
                docs[middle_idx:] + docs[:middle_idx]
            )
            orderings["relevant_in_middle"] = middle_order

        results: dict[str, dict[str, Any]] = {}
        prompt_template = PROMPT_TEMPLATES["structured"]

        for order_name, ordered_docs in orderings.items():
            print(f"\n    ── Ordering: '{order_name}' {'─' * max(1, 35 - len(order_name))}")

            context = self._format_context(ordered_docs)
            order_sources = [d.metadata.get("source", f"doc_{i}")
                             for i, d in enumerate(ordered_docs)]
            print(f"      Doc order: {order_sources}")

            prompt = prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            results[order_name] = {
                "answer": answer,
                "word_count": len(answer.split()),
                "context_order": order_sources,
            }

            print(f"      Words: {len(answer.split())}")
            print(f"      Answer:")
            self._wrap_print(answer, indent=8, width=62)

        # Compare answers for consistency
        print(f"\n    ── Consistency Analysis ──")
        answers = [info["answer"] for info in results.values()]
        ordering_names = list(results.keys())

        # Pairwise word-level Jaccard similarity
        print(f"      Pairwise answer similarity (Jaccard on words):")
        sensitive = False
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                words_i = set(answers[i].lower().split())
                words_j = set(answers[j].lower().split())
                overlap = len(words_i & words_j)
                total = len(words_i | words_j)
                jaccard = overlap / total if total > 0 else 1.0
                status = "✅" if jaccard > 0.6 else "⚠️"
                if jaccard <= 0.6:
                    sensitive = True
                print(f"        {status} {ordering_names[i]} ↔ {ordering_names[j]}: "
                      f"{jaccard:.3f}")

        if sensitive:
            print(f"\n      🚨 Answer is SENSITIVE to context ordering!")
            print("      → The LLM may be affected by 'lost in the middle' effect.")
            print("      → Recommendations:")
            print("        1. Place most relevant docs first AND last")
            print("        2. Use smaller context with fewer, higher-quality chunks")
            print("        3. Use map-reduce instead of context stuffing")
        else:
            print(f"\n      ✅ Answer is ROBUST to context ordering changes.")
            print("      → Good! The model is attending to all parts of the context.")

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Run all generation debugging tools with sample data."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        Generation Debugger — Diagnostic Toolkit                ║")
    print("║        5 tools for understanding generation quality            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        debugger = GenerationDebugger()
    except EnvironmentError as e:
        print(f"\n❌ Setup Error: {e}")
        sys.exit(1)

    # ── Sample data ──────────────────────────────────────────
    sample_question = "What is ACME's annual leave policy?"
    sample_context = (
        "Full-time employees are entitled to 15 days of paid annual leave per calendar year. "
        "Part-time employees receive prorated leave based on their contracted hours. "
        "Annual leave accrues at a rate of 1.25 days per month and may be carried over "
        "up to a maximum of 5 days into the following year. "
        "Unused leave beyond the carryover limit will be forfeited on December 31st. "
        "Employees must submit leave requests at least 14 days in advance through the HR portal."
    )

    sample_docs = [
        Document(
            page_content="Full-time employees are entitled to 15 days of paid annual leave per calendar year. Part-time employees receive prorated leave based on their contracted hours.",
            metadata={"source": "policy_sec2", "section": "annual_leave"},
        ),
        Document(
            page_content="Annual leave accrues at a rate of 1.25 days per month and may be carried over up to a maximum of 5 days into the following year.",
            metadata={"source": "policy_sec2", "section": "annual_leave"},
        ),
        Document(
            page_content="Employees must submit leave requests at least 14 days in advance through the HR portal. Managers must approve or deny requests within 3 business days.",
            metadata={"source": "policy_sec2", "section": "annual_leave"},
        ),
        Document(
            page_content="All employees are entitled to 10 days of paid sick leave per year. Sick leave does not accrue and resets on January 1st each year.",
            metadata={"source": "policy_sec3", "section": "sick_leave"},
        ),
    ]

    # ── Tool 1: Prompt Variations ────────────────────────────
    print("\n\n🔧 TOOL 1: Test Prompt Variations")
    print("=" * 60)
    try:
        debugger.test_prompt_variations(
            question=sample_question,
            context=sample_context,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 2: Temperature Impact ───────────────────────────
    print("\n\n🔧 TOOL 2: Test Temperature Impact")
    print("=" * 60)
    try:
        debugger.test_temperature_impact(
            question=sample_question,
            context=sample_context,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 3: Detect Hallucination ─────────────────────────
    print("\n\n🔧 TOOL 3: Detect Hallucination")
    print("=" * 60)
    # Generate a potentially hallucinated answer
    hallucinated_answer = (
        "ACME provides 15 days of annual leave for full-time employees. "
        "Leave accrues monthly at 1.25 days. Up to 5 days can be carried over. "
        "ACME also offers an additional 5 personal days and a wellness stipend "
        "of $500 per year for gym memberships."
    )
    print(f"    Testing with a mixed answer (some grounded, some fabricated):")
    try:
        debugger.detect_hallucination(
            answer=hallucinated_answer,
            context=sample_context,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 4: Measure Answer Quality ───────────────────────
    print("\n\n🔧 TOOL 4: Measure Answer Quality (LLM-as-Judge)")
    print("=" * 60)
    good_answer = (
        "ACME provides 15 days of paid annual leave per year for full-time employees. "
        "Part-time employees receive prorated leave. Leave accrues at 1.25 days per month, "
        "and up to 5 days can be carried over to the next year. Unused leave beyond the "
        "carryover limit is forfeited on December 31st. Leave requests must be submitted "
        "at least 14 days in advance via the HR portal."
    )
    try:
        debugger.measure_answer_quality(
            question=sample_question,
            answer=good_answer,
            context=sample_context,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    # ── Tool 5: Context Ordering ─────────────────────────────
    print("\n\n🔧 TOOL 5: Test Context Ordering")
    print("=" * 60)
    try:
        debugger.test_context_ordering(
            question=sample_question,
            docs=sample_docs,
        )
    except Exception as e:
        print(f"    ❌ Error: {e}")

    print(f"\n{'═' * 60}")
    print("✅ All generation debugging tools complete.")
    print("   Review output above to identify generation quality issues.")


if __name__ == "__main__":
    main()
