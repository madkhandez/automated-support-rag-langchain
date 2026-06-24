"""
Part 4, File 19: Cost Optimisation — Controlling RAG Spend at Scale

Demonstrates practical strategies for reducing the cost of a production RAG
pipeline without sacrificing answer quality:

1. **Token tracking by operation** — separate accounting for embeddings,
   generation, and (optional) reranking so you know where the money goes.

2. **Adaptive model selection** — route simple questions to cheaper models
   (gpt-4o-mini) and only invoke expensive models (gpt-4o) for complex ones.

3. **Batch processing** — group semantically similar questions so they share
   retrieved context, cutting retrieval and embedding costs.

4. **Cost projection** — real-time per-query, daily, and monthly projections.

5. **Optimisation recommendations** — data-driven suggestions based on actual
   usage patterns (e.g. "switch to gpt-4o-mini for 60 % of queries").

Key concepts:
- Token-level cost accounting
- Complexity-based model routing
- Batch deduplication with cosine similarity
- Projected cost modelling
"""

from __future__ import annotations

import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Load environment ────────────────────────────────────────────────
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pricing table (per 1 000 tokens, USD, Jun 2025)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":                  {"prompt": 0.0025,  "completion": 0.0100},
    "gpt-4o-mini":             {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo":             {"prompt": 0.0100,  "completion": 0.0300},
    "gpt-3.5-turbo":           {"prompt": 0.0005,  "completion": 0.0015},
    "text-embedding-3-small":  {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large":  {"prompt": 0.00013, "completion": 0.0},
}


def _estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    return round(
        prompt_tokens / 1000 * pricing["prompt"]
        + completion_tokens / 1000 * pricing["completion"],
        8,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class OperationCost:
    """Cost record for a single API operation."""

    operation: str           # embedding | generation | reranking
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class QueryCostRecord:
    """Aggregated cost data for one user query."""

    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    model_used: str = ""
    complexity: str = "simple"
    operations: list[OperationCost] = field(default_factory=list)
    total_cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cost Optimiser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CostOptimizer:
    """
    Tracks, analyses, and optimises the cost of a RAG pipeline.

    Main responsibilities:
    * Record per-operation token usage and cost
    * Select models adaptively based on query complexity
    * Batch similar questions to share retrieval context
    * Project daily / monthly costs
    * Recommend optimisation strategies
    """

    # ── complexity keywords (lower-case) ─────────────────────────────
    COMPLEX_SIGNALS: set[str] = {
        "compare", "contrast", "analyse", "analyze", "evaluate", "explain why",
        "step by step", "in detail", "comprehensive", "trade-off", "tradeoff",
        "pros and cons", "advantages and disadvantages", "differences between",
        "how does", "what are the implications", "critically",
        "multi-step", "complex", "advanced", "nuanced",
    }

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        self._records: list[QueryCostRecord] = []
        self._op_costs: list[OperationCost] = []

        # Pre-configured model choices
        self._simple_model = "gpt-4o-mini"
        self._complex_model = os.getenv("LLM_MODEL", "gpt-4o")

        print("  ✅ CostOptimizer initialised")
        print(f"     Simple  model : {self._simple_model}")
        print(f"     Complex model : {self._complex_model}")

    # ── 1. Model selection ───────────────────────────────────────────

    def classify_complexity(self, question: str) -> str:
        """
        Classify a question as ``'simple'`` or ``'complex'``.

        Uses keyword heuristics — cheap and instant.  A production system
        could replace this with a lightweight classifier.
        """
        q_lower = question.lower()

        # Length heuristic: very long questions are often complex
        if len(question.split()) > 25:
            return "complex"

        # Keyword heuristic
        for signal in self.COMPLEX_SIGNALS:
            if signal in q_lower:
                return "complex"

        # Multi-part questions (contains "and" linking two question words)
        question_words = {"what", "how", "why", "when", "where", "which", "who"}
        words = q_lower.split()
        qw_count = sum(1 for w in words if w in question_words)
        if qw_count >= 2:
            return "complex"

        return "simple"

    def model_selector(self, query_complexity: str) -> str:
        """
        Return the most cost-effective model for the given complexity.

        * ``'simple'`` → gpt-4o-mini  (≈ 17× cheaper)
        * ``'complex'`` → gpt-4o
        """
        if query_complexity == "complex":
            return self._complex_model
        return self._simple_model

    def select_model_for_query(self, question: str) -> tuple[str, str]:
        """Convenience: classify + select in one call.  Returns (model, complexity)."""
        complexity = self.classify_complexity(question)
        model = self.model_selector(complexity)
        return model, complexity

    # ── 2. Token tracking ────────────────────────────────────────────

    def track_operation(
        self,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> OperationCost:
        """Record a single API operation's token usage and cost."""
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        op = OperationCost(
            operation=operation,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        self._op_costs.append(op)
        return op

    def track_query(
        self,
        question: str,
        model: str,
        complexity: str,
        operations: list[OperationCost],
        latency_ms: float = 0.0,
    ) -> QueryCostRecord:
        """Record a full query's cost (sum of operations)."""
        total = sum(op.cost_usd for op in operations)
        rec = QueryCostRecord(
            question=question,
            model_used=model,
            complexity=complexity,
            operations=operations,
            total_cost=round(total, 8),
            latency_ms=latency_ms,
        )
        self._records.append(rec)
        return rec

    # ── 3. Batch processing ──────────────────────────────────────────

    def batch_processing_optimizer(
        self, questions: list[str], similarity_threshold: float = 0.85,
    ) -> list[list[str]]:
        """
        Group *questions* into batches where members are semantically similar.

        Questions within the same batch can share retrieved context,
        saving both embedding and retrieval cost.

        Returns a list of groups (each group is a list of questions).
        """
        if not questions:
            return []
        if len(questions) == 1:
            return [questions]

        embeddings = self.embeddings.embed_documents(questions)
        embs = np.array(embeddings, dtype=np.float64)

        # Normalise
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embs / norms

        # Cosine similarity matrix
        sim_matrix = normed @ normed.T

        # Greedy clustering
        assigned: set[int] = set()
        groups: list[list[int]] = []
        for i in range(len(questions)):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(questions)):
                if j in assigned:
                    continue
                if sim_matrix[i, j] >= similarity_threshold:
                    group.append(j)
                    assigned.add(j)
            groups.append(group)

        return [[questions[idx] for idx in g] for g in groups]

    # ── 4. Cost analytics ────────────────────────────────────────────

    def cost_per_query(self) -> float:
        """Average cost (USD) per query."""
        if not self._records:
            return 0.0
        return round(
            sum(r.total_cost for r in self._records) / len(self._records), 8
        )

    def cost_breakdown_by_operation(self) -> dict[str, float]:
        """Total cost grouped by operation type."""
        breakdown: dict[str, float] = defaultdict(float)
        for op in self._op_costs:
            breakdown[op.operation] += op.cost_usd
        return {k: round(v, 6) for k, v in sorted(breakdown.items())}

    def cost_breakdown_by_model(self) -> dict[str, float]:
        """Total cost grouped by model."""
        breakdown: dict[str, float] = defaultdict(float)
        for op in self._op_costs:
            breakdown[op.model] += op.cost_usd
        return {k: round(v, 6) for k, v in sorted(breakdown.items())}

    def projected_daily_cost(self, queries_per_day: int | None = None) -> float:
        """Project daily spend based on current average cost per query."""
        qpd = queries_per_day or max(len(self._records), 1) * 24  # naive linear
        return round(self.cost_per_query() * qpd, 6)

    def projected_monthly_cost(self, queries_per_day: int | None = None) -> float:
        """Project 30-day spend."""
        return round(self.projected_daily_cost(queries_per_day) * 30, 4)

    # ── 5. Optimisation recommendations ──────────────────────────────

    def recommend_optimizations(self) -> list[str]:
        """
        Analyse usage patterns and return actionable optimisation tips.
        """
        tips: list[str] = []
        if not self._records:
            return ["Not enough data to make recommendations yet."]

        # 1. Model downgrade opportunity
        complex_count = sum(1 for r in self._records if r.complexity == "complex")
        simple_count = len(self._records) - complex_count
        complex_pct = complex_count / len(self._records) * 100
        if complex_pct > 50:
            tips.append(
                f"🔍 {complex_pct:.0f}% of queries routed to the expensive model. "
                f"Review complexity heuristics — some 'complex' questions may be "
                f"answerable by {self._simple_model}."
            )
        if simple_count > 0:
            saved = simple_count * (_estimate_cost(self._complex_model, 600, 200)
                                     - _estimate_cost(self._simple_model, 600, 200))
            tips.append(
                f"💰 Model routing saved ~${saved:.4f} by sending {simple_count} "
                f"simple queries to {self._simple_model}."
            )

        # 2. Embedding cost share
        emb_cost = sum(op.cost_usd for op in self._op_costs if op.operation == "embedding")
        total_cost = sum(op.cost_usd for op in self._op_costs) or 1
        emb_pct = emb_cost / total_cost * 100
        if emb_pct > 40:
            tips.append(
                f"📦 Embeddings account for {emb_pct:.0f}% of total cost. "
                f"Enable the InMemoryEmbeddingCache (file 17) to cut this "
                f"by 50–90%."
            )

        # 3. Batch optimisation
        if len(self._records) >= 10:
            tips.append(
                "🔄 For bulk workloads, use batch_processing_optimizer() to group "
                "similar questions and share retrieval context."
            )

        # 4. Caching
        tips.append(
            "⚡ Enable semantic caching (file 17) with threshold=0.95 to "
            "skip generation for repeated / rephrased questions."
        )

        # 5. Smaller embedding model
        if os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") == "text-embedding-3-large":
            tips.append(
                "📐 Consider switching to text-embedding-3-small — 6.5× cheaper "
                "with comparable retrieval quality for most use cases."
            )

        return tips

    # ── 6. Dashboard ─────────────────────────────────────────────────

    def print_cost_dashboard(self) -> None:
        """Render a formatted cost dashboard to stdout."""
        cpq = self.cost_per_query()
        by_op = self.cost_breakdown_by_operation()
        by_model = self.cost_breakdown_by_model()
        total = sum(r.total_cost for r in self._records)
        daily = self.projected_daily_cost(queries_per_day=1000)
        monthly = self.projected_monthly_cost(queries_per_day=1000)

        complexity_dist: dict[str, int] = defaultdict(int)
        for r in self._records:
            complexity_dist[r.complexity] += 1

        print()
        print("╔" + "═" * 62 + "╗")
        print("║" + "  💰  Cost Optimisation Dashboard".ljust(62) + "║")
        print("╠" + "═" * 62 + "╣")

        print("║" + "  SUMMARY".ljust(62) + "║")
        print("║" + f"    Total queries         : {len(self._records)}".ljust(62) + "║")
        print("║" + f"    Total cost            : ${total:.6f}".ljust(62) + "║")
        print("║" + f"    Avg cost / query      : ${cpq:.6f}".ljust(62) + "║")
        print("║" + f"    Projected daily (1k)  : ${daily:.4f}".ljust(62) + "║")
        print("║" + f"    Projected monthly(1k) : ${monthly:.4f}".ljust(62) + "║")
        print("╠" + "═" * 62 + "╣")

        print("║" + "  COST BY OPERATION".ljust(62) + "║")
        for op, cost in by_op.items():
            pct = cost / total * 100 if total else 0
            bar = "█" * int(pct / 2)
            print("║" + f"    {op:<14} ${cost:.6f}  ({pct:5.1f}%) {bar}".ljust(62) + "║")
        print("╠" + "═" * 62 + "╣")

        print("║" + "  COST BY MODEL".ljust(62) + "║")
        for mdl, cost in by_model.items():
            pct = cost / total * 100 if total else 0
            print("║" + f"    {mdl:<26} ${cost:.6f}  ({pct:5.1f}%)".ljust(62) + "║")
        print("╠" + "═" * 62 + "╣")

        print("║" + "  QUERY COMPLEXITY DISTRIBUTION".ljust(62) + "║")
        for comp, cnt in sorted(complexity_dist.items()):
            pct = cnt / len(self._records) * 100 if self._records else 0
            bar = "█" * int(pct / 2)
            print("║" + f"    {comp:<10} {cnt:>4} queries ({pct:5.1f}%) {bar}".ljust(62) + "║")
        print("╠" + "═" * 62 + "╣")

        print("║" + "  OPTIMISATION RECOMMENDATIONS".ljust(62) + "║")
        for tip in self.recommend_optimizations():
            # Wrap long tips
            words = tip.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 60:
                    print("║" + line.ljust(62) + "║")
                    line = "      " + w
                else:
                    line += (" " if line.strip() else "") + w
            if line.strip():
                print("║" + line.ljust(62) + "║")
            print("║" + " ".ljust(62) + "║")

        print("╚" + "═" * 62 + "╝")
        print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _simulate_queries(optimizer: CostOptimizer) -> None:
    """Feed a realistic mix of queries through the optimizer."""
    import random

    questions = [
        # Simple
        "What is RAG?",
        "Define vector embeddings.",
        "What is ChromaDB?",
        "Explain cosine similarity.",
        "What is LangChain?",
        "How does chunking work?",
        "What is an embedding model?",
        # Complex
        "Compare and contrast pgvector vs. Pinecone for production deployments.",
        "Explain step by step how a RAG pipeline retrieves and generates answers.",
        "What are the advantages and disadvantages of different chunking strategies?",
        "Analyse the trade-offs between embedding model size and retrieval quality.",
        "How does the choice of vector index (IVF vs. HNSW) affect latency and recall?",
    ]

    random.seed(42)
    for q in questions:
        model, complexity = optimizer.select_model_for_query(q)

        # Simulate token counts
        prompt_tokens = random.randint(300, 1200)
        completion_tokens = random.randint(80, 500)
        embed_tokens = random.randint(50, 200)

        ops: list[OperationCost] = []

        # Embedding operation
        ops.append(optimizer.track_operation(
            "embedding", "text-embedding-3-small", embed_tokens
        ))

        # Generation operation
        ops.append(optimizer.track_operation(
            "generation", model, prompt_tokens, completion_tokens
        ))

        # Optional reranking (20 % of the time)
        if random.random() < 0.2:
            rerank_tokens = random.randint(100, 400)
            ops.append(optimizer.track_operation(
                "reranking", "gpt-4o-mini", rerank_tokens
            ))

        latency = random.uniform(400, 2500)
        rec = optimizer.track_query(q, model, complexity, ops, latency)

        print(f"  [{complexity:>7}] {model:<14}  ${rec.total_cost:.6f}  "
              f"│ {q[:55]}")


def main() -> None:
    """Demonstrate cost tracking, model routing, batching, and recommendations."""

    print("=" * 70)
    print("  Part 4 · File 19 — Cost Optimisation")
    print("=" * 70)

    # ── 1. Initialise optimiser ──────────────────────────────────────
    print("\n🔧 Step 1: Initialising CostOptimizer …\n")
    optimizer = CostOptimizer()

    # ── 2. Complexity classification demo ────────────────────────────
    print("\n📝 Step 2: Query complexity classification\n")
    demo_questions = [
        "What is RAG?",
        "Compare and contrast HNSW vs. IVF-Flat for million-scale datasets.",
        "Define cosine similarity.",
        "Explain step by step how attention works in transformer models.",
        "What is LangChain?",
    ]
    for q in demo_questions:
        model, complexity = optimizer.select_model_for_query(q)
        print(f"  [{complexity:>7}] → {model:<14}  │ {q[:58]}")

    # ── 3. Batch processing demo ─────────────────────────────────────
    print("\n\n📦 Step 3: Batch processing optimisation\n")
    batch_questions = [
        "What is RAG?",
        "How does RAG work?",
        "Explain retrieval augmented generation",
        "What is a vector database?",
        "How do vector databases store embeddings?",
        "What is the company leave policy?",
    ]
    print("  Input questions:")
    for q in batch_questions:
        print(f"    • {q}")

    groups = optimizer.batch_processing_optimizer(batch_questions, similarity_threshold=0.80)
    print(f"\n  Grouped into {len(groups)} batches:")
    for i, group in enumerate(groups, 1):
        print(f"    Batch {i}:")
        for q in group:
            print(f"      ‣ {q}")

    naive_cost = len(batch_questions) * 0.003  # hypothetical per-query cost
    batched_cost = len(groups) * 0.003
    saving = naive_cost - batched_cost
    print(f"\n  Estimated saving: ${saving:.4f} "
          f"({saving / naive_cost * 100:.0f}% reduction)")

    # ── 4. Simulate queries ──────────────────────────────────────────
    print("\n\n🔄 Step 4: Simulating 12 queries with cost tracking\n")
    _simulate_queries(optimizer)

    # ── 5. Cost analytics ────────────────────────────────────────────
    print("\n\n📊 Step 5: Cost analytics\n")
    print(f"  Avg cost per query    : ${optimizer.cost_per_query():.6f}")
    print(f"  Projected daily (1k)  : ${optimizer.projected_daily_cost(1000):.4f}")
    print(f"  Projected monthly (1k): ${optimizer.projected_monthly_cost(1000):.4f}")

    print("\n  Cost by operation:")
    for op, cost in optimizer.cost_breakdown_by_operation().items():
        print(f"    {op:<14}: ${cost:.6f}")

    print("\n  Cost by model:")
    for mdl, cost in optimizer.cost_breakdown_by_model().items():
        print(f"    {mdl:<26}: ${cost:.6f}")

    # ── 6. Dashboard ─────────────────────────────────────────────────
    print("\n📊 Step 6: Full cost dashboard")
    optimizer.print_cost_dashboard()

    print("=" * 70)
    print("  ✅ Cost optimisation demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
