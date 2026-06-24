"""
Part 4, File 18: Monitoring — Production Observability for RAG Systems

Implements a ProductionMonitor that tracks per-query metrics and provides
real-time alerting across three production visibility pillars:

  1. **Latency alerting**   — warns when a query exceeds 3 000 ms
  2. **Quality alerting**   — warns when average retrieval score drops below 0.7
  3. **Cost alerting**      — warns when projected daily cost exceeds $10

All metrics are persisted to a JSON-lines log file for later analysis and
a pretty-printed dashboard summarises the current state of the system.

Key concepts:
- Structured metric logging (JSON-lines)
- Statistical aggregations (mean, p95)
- Threshold-based alerting
- Dashboard visualisation in the terminal
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────────
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class QueryMetrics:
    """Metrics captured for a single RAG query."""

    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    question: str = ""
    latency_ms: float = 0.0
    retrieval_count: int = 0
    retrieval_score: float = 0.0       # average similarity of retrieved docs
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0        # USD
    model: str = ""
    cache_hit: bool = False
    error: str | None = None

    @property
    def token_usage(self) -> int:
        return self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlertEvent:
    """A triggered alert."""

    alert_type: str          # latency | quality | cost
    severity: str            # warning | critical
    message: str
    value: float
    threshold: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cost estimator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prices per 1 000 tokens (USD) — updated Jun 2025
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":              {"prompt": 0.0025,  "completion": 0.0100},
    "gpt-4o-mini":         {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo":         {"prompt": 0.0100,  "completion": 0.0300},
    "gpt-3.5-turbo":       {"prompt": 0.0005,  "completion": 0.0015},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the cost (USD) of a single API call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    cost = (
        prompt_tokens / 1000 * pricing["prompt"]
        + completion_tokens / 1000 * pricing["completion"]
    )
    return round(cost, 6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Production Monitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ProductionMonitor:
    """
    Centralised monitoring for a RAG system.

    Collects per-query metrics, persists them to a JSON-lines file,
    evaluates alert rules, and renders a terminal dashboard.
    """

    # ── Alert thresholds ─────────────────────────────────────────────
    LATENCY_WARN_MS: float = 3_000.0
    QUALITY_WARN_SCORE: float = 0.7
    DAILY_COST_WARN_USD: float = 10.0

    def __init__(self, log_file: str | Path | None = None) -> None:
        self.log_file = Path(
            log_file
            or Path(__file__).resolve().parent.parent
            / "logs"
            / "rag_metrics.jsonl"
        )
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._metrics: list[QueryMetrics] = []
        self._alerts: list[AlertEvent] = []

        # Load existing metrics from the log file (if present)
        self._load_existing_metrics()

        print(f"  📋 ProductionMonitor initialised — log: {self.log_file}")

    # ── persistence ──────────────────────────────────────────────────

    def _load_existing_metrics(self) -> None:
        """Read back previously persisted metrics."""
        if not self.log_file.exists():
            return
        try:
            with open(self.log_file, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._metrics.append(QueryMetrics(**data))
        except Exception:
            pass  # graceful — a corrupt log shouldn't crash the app

    def _append_to_log(self, metrics: QueryMetrics) -> None:
        with open(self.log_file, "a") as fh:
            fh.write(json.dumps(metrics.to_dict()) + "\n")

    # ── public API ───────────────────────────────────────────────────

    def log_query_metrics(self, metrics: dict[str, Any] | QueryMetrics) -> QueryMetrics:
        """
        Record a query's metrics, persist to disk, and evaluate alerts.

        Accepts either a QueryMetrics instance or a plain dict.
        """
        if isinstance(metrics, dict):
            qm = QueryMetrics(**{
                k: v for k, v in metrics.items()
                if k in QueryMetrics.__dataclass_fields__
            })
        else:
            qm = metrics

        # Auto-compute cost if not supplied
        if qm.estimated_cost == 0.0 and qm.total_tokens > 0:
            qm.estimated_cost = estimate_cost(
                qm.model or "gpt-4o", qm.prompt_tokens, qm.completion_tokens
            )

        self._metrics.append(qm)
        self._append_to_log(qm)
        self._evaluate_alerts(qm)
        return qm

    def _evaluate_alerts(self, qm: QueryMetrics) -> None:
        """Check alert thresholds against the latest metric."""

        # 1. Latency
        if qm.latency_ms > self.LATENCY_WARN_MS:
            alert = AlertEvent(
                alert_type="latency",
                severity="warning",
                message=f"Query {qm.query_id} took {qm.latency_ms:.0f} ms "
                        f"(threshold {self.LATENCY_WARN_MS:.0f} ms)",
                value=qm.latency_ms,
                threshold=self.LATENCY_WARN_MS,
            )
            self._alerts.append(alert)
            print(f"  ⚠️  ALERT [{alert.severity}] {alert.message}")

        # 2. Quality (rolling average over last 20 queries)
        recent_scores = [
            m.retrieval_score
            for m in self._metrics[-20:]
            if m.retrieval_score > 0
        ]
        if recent_scores:
            avg_score = statistics.mean(recent_scores)
            if avg_score < self.QUALITY_WARN_SCORE:
                alert = AlertEvent(
                    alert_type="quality",
                    severity="warning",
                    message=f"Avg retrieval score {avg_score:.3f} < "
                            f"{self.QUALITY_WARN_SCORE} over last {len(recent_scores)} queries",
                    value=avg_score,
                    threshold=self.QUALITY_WARN_SCORE,
                )
                self._alerts.append(alert)
                print(f"  ⚠️  ALERT [{alert.severity}] {alert.message}")

        # 3. Daily cost
        today = datetime.now(timezone.utc).date().isoformat()
        daily_cost = sum(
            m.estimated_cost
            for m in self._metrics
            if m.timestamp.startswith(today)
        )
        if daily_cost > self.DAILY_COST_WARN_USD:
            alert = AlertEvent(
                alert_type="cost",
                severity="warning",
                message=f"Daily cost ${daily_cost:.4f} exceeds "
                        f"${self.DAILY_COST_WARN_USD:.2f} threshold",
                value=daily_cost,
                threshold=self.DAILY_COST_WARN_USD,
            )
            self._alerts.append(alert)
            print(f"  ⚠️  ALERT [{alert.severity}] {alert.message}")

    # ── aggregation ──────────────────────────────────────────────────

    def get_dashboard_stats(self) -> dict[str, Any]:
        """
        Compute aggregate statistics over all recorded metrics.

        Returns
        -------
        dict with keys: avg_latency, p95_latency, total_queries,
                        total_cost, avg_retrieval_score, total_tokens,
                        cache_hit_rate, alerts_triggered
        """
        if not self._metrics:
            return {
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_retrieval_score": 0.0,
                "total_tokens": 0,
                "cache_hit_rate": 0.0,
                "alerts_triggered": 0,
            }

        latencies = [m.latency_ms for m in self._metrics]
        scores = [m.retrieval_score for m in self._metrics if m.retrieval_score > 0]
        total_cost = sum(m.estimated_cost for m in self._metrics)
        total_tokens = sum(m.total_tokens for m in self._metrics)
        cache_hits = sum(1 for m in self._metrics if m.cache_hit)

        # p95 latency
        sorted_lat = sorted(latencies)
        p95_idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
        p95_latency = sorted_lat[max(p95_idx, 0)]

        return {
            "avg_latency": round(statistics.mean(latencies), 2),
            "p95_latency": round(p95_latency, 2),
            "total_queries": len(self._metrics),
            "total_cost": round(total_cost, 6),
            "avg_retrieval_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "total_tokens": total_tokens,
            "cache_hit_rate": round(cache_hits / len(self._metrics), 4),
            "alerts_triggered": len(self._alerts),
        }

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Break costs down by model."""
        by_model: dict[str, float] = {}
        for m in self._metrics:
            model = m.model or "unknown"
            by_model[model] = by_model.get(model, 0.0) + m.estimated_cost
        return {k: round(v, 6) for k, v in sorted(by_model.items())}

    def get_recent_alerts(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last *n* alerts as dicts."""
        return [
            {
                "type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "value": a.value,
                "threshold": a.threshold,
                "timestamp": a.timestamp,
            }
            for a in self._alerts[-n:]
        ]

    # ── dashboard rendering ──────────────────────────────────────────

    def print_dashboard(self) -> None:
        """Render a nicely-formatted monitoring dashboard to stdout."""
        stats = self.get_dashboard_stats()
        cost_by_model = self.get_cost_breakdown()

        print()
        print("╔" + "═" * 60 + "╗")
        print("║" + "  📊  RAG Production Monitoring Dashboard".ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

        # ── Overview
        print("║" + "  OVERVIEW".ljust(60) + "║")
        print("║" + f"    Total queries       : {stats['total_queries']:<20}".ljust(60) + "║")
        print("║" + f"    Total tokens        : {stats['total_tokens']:,}".ljust(60) + "║")
        print("║" + f"    Cache hit rate      : {stats['cache_hit_rate']:.1%}".ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

        # ── Latency
        avg_lat = stats["avg_latency"]
        p95_lat = stats["p95_latency"]
        lat_status = "🟢" if p95_lat < self.LATENCY_WARN_MS else "🔴"
        print("║" + f"  LATENCY  {lat_status}".ljust(60) + "║")
        print("║" + f"    Average             : {avg_lat:,.1f} ms".ljust(60) + "║")
        print("║" + f"    P95                 : {p95_lat:,.1f} ms".ljust(60) + "║")
        print("║" + f"    Threshold (warn)    : {self.LATENCY_WARN_MS:,.0f} ms".ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

        # ── Quality
        avg_score = stats["avg_retrieval_score"]
        q_status = "🟢" if avg_score >= self.QUALITY_WARN_SCORE else "🔴"
        print("║" + f"  RETRIEVAL QUALITY  {q_status}".ljust(60) + "║")
        print("║" + f"    Avg retrieval score  : {avg_score:.4f}".ljust(60) + "║")
        print("║" + f"    Threshold (warn)     : {self.QUALITY_WARN_SCORE}".ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

        # ── Cost
        total_cost = stats["total_cost"]
        today = datetime.now(timezone.utc).date().isoformat()
        daily_cost = sum(
            m.estimated_cost for m in self._metrics if m.timestamp.startswith(today)
        )
        cost_status = "🟢" if daily_cost < self.DAILY_COST_WARN_USD else "🔴"
        print("║" + f"  COST  {cost_status}".ljust(60) + "║")
        print("║" + f"    Total cost           : ${total_cost:.6f}".ljust(60) + "║")
        print("║" + f"    Today's cost         : ${daily_cost:.6f}".ljust(60) + "║")
        print("║" + f"    Daily threshold      : ${self.DAILY_COST_WARN_USD:.2f}".ljust(60) + "║")
        if cost_by_model:
            print("║" + "    By model:".ljust(60) + "║")
            for model, cost in cost_by_model.items():
                print("║" + f"      {model:<24}: ${cost:.6f}".ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

        # ── Alerts
        recent = self.get_recent_alerts(5)
        print("║" + f"  RECENT ALERTS ({len(self._alerts)} total)".ljust(60) + "║")
        if recent:
            for a in recent:
                line = f"    [{a['severity']}] {a['type']}: {a['message'][:42]}"
                print("║" + line.ljust(60) + "║")
        else:
            print("║" + "    No alerts triggered ✅".ljust(60) + "║")

        print("╚" + "═" * 60 + "╝")
        print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Simulate a stream of RAG queries and display the monitoring dashboard."""

    print("=" * 70)
    print("  Part 4 · File 18 — Production Monitoring")
    print("=" * 70)

    # Use a separate log file so we don't pollute real data
    log_path = (
        Path(__file__).resolve().parent.parent / "logs" / "demo_metrics.jsonl"
    )
    # Clear previous demo log
    if log_path.exists():
        log_path.unlink()

    monitor = ProductionMonitor(log_file=log_path)

    # ── Simulate 25 queries with realistic variation ─────────────────
    print("\n📡 Simulating 25 RAG queries …\n")

    sample_questions = [
        "What is RAG?",
        "Explain vector embeddings",
        "How does pgvector work?",
        "What is LangChain?",
        "Company leave policy details",
        "How to implement caching?",
        "What are transformers?",
        "Explain cosine similarity",
        "How does chunking affect retrieval?",
        "What is a retrieval score?",
    ]

    models = ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"]  # mostly mini

    for i in range(25):
        question = sample_questions[i % len(sample_questions)]
        model = random.choice(models)
        is_cache_hit = random.random() < 0.3  # 30 % cache hits

        # Simulate realistic metric values
        if is_cache_hit:
            latency = random.uniform(1, 15)
            prompt_tokens = 0
            completion_tokens = 0
        else:
            latency = random.uniform(400, 2800)
            prompt_tokens = random.randint(200, 1200)
            completion_tokens = random.randint(50, 400)

        # Occasionally simulate a slow query to trigger latency alert
        if i == 18:
            latency = 4500.0

        retrieval_score = random.uniform(0.65, 0.95)
        # Simulate a brief quality dip
        if 10 <= i <= 14:
            retrieval_score = random.uniform(0.45, 0.65)

        metrics = QueryMetrics(
            question=question,
            latency_ms=round(latency, 2),
            retrieval_count=random.randint(2, 6),
            retrieval_score=round(retrieval_score, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            cache_hit=is_cache_hit,
        )

        monitor.log_query_metrics(metrics)
        status = "⚡ cache" if is_cache_hit else "🔍 full"
        print(f"  [{i + 1:>2}] {status}  {latency:>7.1f} ms  "
              f"score={retrieval_score:.2f}  model={model}")

    # ── Dashboard ────────────────────────────────────────────────────
    print("\n\n📊 Final dashboard:")
    monitor.print_dashboard()

    # ── Raw stats ────────────────────────────────────────────────────
    print("Raw dashboard stats:")
    stats = monitor.get_dashboard_stats()
    for k, v in stats.items():
        print(f"  {k:>22s}: {v}")

    print(f"\n  📁 Metrics log: {monitor.log_file}")
    print("\n" + "=" * 70)
    print("  ✅ Monitoring demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
