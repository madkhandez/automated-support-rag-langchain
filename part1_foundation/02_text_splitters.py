"""
02_text_splitters.py — Text Splitting Strategies for RAG

Demonstrates how to break large documents into smaller, semantically
meaningful chunks suitable for embedding and retrieval. Covers:
  • RecursiveCharacterTextSplitter (recommended default)
  • TokenTextSplitter (tiktoken-based, precise token control)
  • CharacterTextSplitter (simple baseline)

Key design decisions:
  - chunk_size controls retrieval precision vs. context richness
  - chunk_overlap ensures no information is lost at boundaries
  - Recursive splitting preserves paragraph/sentence structure

Run:
    python part1_foundation/02_text_splitters.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# ── Resolve project paths ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"

load_dotenv(PROJECT_ROOT / ".env")


# ════════════════════════════════════════════════════════════════════
# Helper: Load a sample document
# ════════════════════════════════════════════════════════════════════
def _load_sample_document() -> Document:
    """Load the company policy text file as a single Document."""
    filepath = DOCS_DIR / "company_policy.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Sample document not found: {filepath}\n"
            "Make sure docs/company_policy.txt exists."
        )
    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()
    return docs[0]


# ════════════════════════════════════════════════════════════════════
# Chunk Size Analyzer
# ════════════════════════════════════════════════════════════════════
def chunk_size_analyzer(
    chunks: list[Document],
    label: str = "Chunks",
) -> dict[str, int | float]:
    """Analyse and print statistics about a list of text chunks.

    Returns a dict with: total, avg_size, min_size, max_size,
    median_size, and semantic_boundary_count.
    """
    if not chunks:
        print(f"  {label}: No chunks to analyse.")
        return {}

    sizes = [len(c.page_content) for c in chunks]

    # Count chunks whose content starts or ends at a semantic boundary
    # (paragraph break, heading, list item, section divider)
    boundary_markers = ("\n\n", "\n---", "\n# ", "\n## ", "\n- ", "\n1. ", "\n2. ")
    semantic_count = 0
    for c in chunks:
        text = c.page_content
        starts_at_boundary = any(text.lstrip().startswith(m.lstrip()) for m in boundary_markers)
        ends_at_boundary = text.rstrip().endswith((".", ":", "\n"))
        if starts_at_boundary or ends_at_boundary:
            semantic_count += 1

    stats = {
        "total": len(chunks),
        "avg_size": int(statistics.mean(sizes)),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "median_size": int(statistics.median(sizes)),
        "semantic_boundary_count": semantic_count,
    }

    print(f"\n  📐  Chunk Analysis: {label}")
    print(f"      Total chunks           : {stats['total']}")
    print(f"      Average chunk size     : {stats['avg_size']:,} chars")
    print(f"      Min chunk size         : {stats['min_size']:,} chars")
    print(f"      Max chunk size         : {stats['max_size']:,} chars")
    print(f"      Median chunk size      : {stats['median_size']:,} chars")
    print(f"      Semantic boundaries    : {stats['semantic_boundary_count']}/{stats['total']} "
          f"({stats['semantic_boundary_count'] / stats['total'] * 100:.0f}%)")

    return stats


# ════════════════════════════════════════════════════════════════════
# TextSplittingPipeline
# ════════════════════════════════════════════════════════════════════
class TextSplittingPipeline:
    """Compare three text splitting strategies side-by-side."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"✂️  TextSplittingPipeline initialised")
        print(f"    chunk_size   = {self.chunk_size}")
        print(f"    chunk_overlap = {self.chunk_overlap}")

    # ── Strategy 1: RecursiveCharacterTextSplitter ──────────────────
    def split_recursive(self, doc: Document) -> list[Document]:
        """Split using RecursiveCharacterTextSplitter.

        This is the recommended default. It tries to split at natural
        boundaries in order: \\n\\n → \\n → ' ' → '' (character level).

        Args:
            doc: A single Document to split.

        Returns:
            List of Document chunks.
        """
        print("\n" + "─" * 55)
        print("  Strategy 1: RecursiveCharacterTextSplitter")
        print("─" * 55)
        print("  Splits at: \\n\\n → \\n → ' ' → '' (character)")
        print("  Best for:  General-purpose document splitting")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,  # Track character offset in metadata
            separators=["\n\n", "\n", " ", ""],
        )

        start = time.perf_counter()
        chunks = splitter.split_documents([doc])
        elapsed = time.perf_counter() - start

        print(f"  Time: {elapsed:.4f}s")
        self._print_chunk_previews(chunks, max_show=3)

        return chunks

    # ── Strategy 2: TokenTextSplitter ───────────────────────────────
    def split_by_tokens(self, doc: Document) -> list[Document]:
        """Split using TokenTextSplitter with tiktoken cl100k_base.

        Ensures each chunk fits within a precise token budget — critical
        for models with fixed context windows and for cost control.

        Args:
            doc: A single Document to split.

        Returns:
            List of Document chunks.
        """
        print("\n" + "─" * 55)
        print("  Strategy 2: TokenTextSplitter (tiktoken cl100k_base)")
        print("─" * 55)
        print("  Splits at: Token boundaries (≈4 chars per token)")
        print("  Best for:  Precise token-budget control")

        # Convert character-based sizes to approximate token counts
        token_chunk_size = self.chunk_size // 4  # ~250 tokens
        token_overlap = self.chunk_overlap // 4   # ~50 tokens

        splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=token_overlap,
            encoding_name="cl100k_base",
        )

        start = time.perf_counter()
        chunks = splitter.split_documents([doc])
        elapsed = time.perf_counter() - start

        print(f"  Token chunk_size : {token_chunk_size} tokens")
        print(f"  Token overlap    : {token_overlap} tokens")
        print(f"  Time: {elapsed:.4f}s")
        self._print_chunk_previews(chunks, max_show=3)

        return chunks

    # ── Strategy 3: CharacterTextSplitter ───────────────────────────
    def split_by_character(self, doc: Document) -> list[Document]:
        """Split using CharacterTextSplitter (simple baseline).

        Splits only on a single separator (default: \\n\\n). This often
        produces chunks larger than chunk_size if the separator is rare.

        Args:
            doc: A single Document to split.

        Returns:
            List of Document chunks.
        """
        print("\n" + "─" * 55)
        print("  Strategy 3: CharacterTextSplitter (baseline)")
        print("─" * 55)
        print("  Splits at: Single separator (\\n\\n)")
        print("  Best for:  Simple documents with clear paragraph breaks")

        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n",
        )

        start = time.perf_counter()
        chunks = splitter.split_documents([doc])
        elapsed = time.perf_counter() - start

        print(f"  Time: {elapsed:.4f}s")
        self._print_chunk_previews(chunks, max_show=3)

        return chunks

    # ── Visual Comparison Table ─────────────────────────────────────
    @staticmethod
    def print_comparison_table(
        results: dict[str, dict[str, int | float]],
    ) -> None:
        """Print a formatted comparison table of splitting strategies.

        Args:
            results: Dict mapping strategy name → chunk_size_analyzer stats.
        """
        print("\n" + "═" * 75)
        print("  📊  SPLITTING STRATEGY COMPARISON TABLE")
        print("═" * 75)

        header = (
            f"  {'Strategy':<30} {'Chunks':>6} {'Avg':>6} "
            f"{'Min':>6} {'Max':>6} {'Semantic':>10}"
        )
        print(header)
        print("  " + "─" * 72)

        for name, stats in results.items():
            if not stats:
                continue
            semantic_pct = (
                f"{stats['semantic_boundary_count']}/{stats['total']}"
            )
            row = (
                f"  {name:<30} {stats['total']:>6} {stats['avg_size']:>6} "
                f"{stats['min_size']:>6} {stats['max_size']:>6} {semantic_pct:>10}"
            )
            print(row)

        print("═" * 75)
        print()
        print("  Legend:")
        print("    Chunks   = total number of chunks produced")
        print("    Avg/Min/Max = chunk size in characters")
        print("    Semantic = chunks starting/ending at semantic boundaries")
        print()

    # ── Helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _print_chunk_previews(chunks: list[Document], max_show: int = 3) -> None:
        """Show previews of the first few chunks."""
        show = min(len(chunks), max_show)
        for i in range(show):
            text = chunks[i].page_content
            preview = text[:90].replace("\n", "\\n")
            meta = chunks[i].metadata
            print(f"\n  Chunk {i + 1}/{len(chunks)} ({len(text):,} chars):")
            print(f"    \"{preview}…\"")
            print(f"    metadata: {meta}")

    # ── Run all strategies ──────────────────────────────────────────
    def run_all(self, doc: Document) -> dict[str, list[Document]]:
        """Run all three splitting strategies and print comparison.

        Args:
            doc: A single Document to split.

        Returns:
            Dict mapping strategy name → list of chunks.
        """
        results: dict[str, list[Document]] = {}
        stats: dict[str, dict[str, int | float]] = {}

        # Run each strategy
        for name, method in [
            ("Recursive (recommended)", self.split_recursive),
            ("Token-based (tiktoken)", self.split_by_tokens),
            ("Character (baseline)", self.split_by_character),
        ]:
            chunks = method(doc)
            results[name] = chunks
            stats[name] = chunk_size_analyzer(chunks, label=name)

        # Print comparison
        self.print_comparison_table(stats)

        return results


# ════════════════════════════════════════════════════════════════════
# Main demonstration
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run all splitting strategies on the sample document."""
    print("=" * 60)
    print("  Part 1.2 — Text Splitting Strategies for RAG")
    print("=" * 60)

    # Load sample document
    print("\n📄  Loading sample document…")
    doc = _load_sample_document()
    print(f"    Source    : {doc.metadata.get('source', 'unknown')}")
    print(f"    Length    : {len(doc.page_content):,} characters")
    print(f"    Preview  : \"{doc.page_content[:80]}…\"")

    # Run pipeline
    pipeline = TextSplittingPipeline(chunk_size=1000, chunk_overlap=200)
    results = pipeline.run_all(doc)

    # ── Show overlap behaviour ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("  BONUS: Understanding Chunk Overlap")
    print("─" * 60)

    recursive_chunks = results.get("Recursive (recommended)", [])
    if len(recursive_chunks) >= 2:
        c1 = recursive_chunks[0].page_content
        c2 = recursive_chunks[1].page_content

        # Find the actual overlap
        overlap_text = ""
        for length in range(min(len(c1), len(c2)), 0, -1):
            if c1.endswith(c2[:length]):
                overlap_text = c2[:length]
                break

        if overlap_text:
            preview = overlap_text[:120].replace("\n", "\\n")
            print(f"  Chunk 1 ends → Chunk 2 begins with {len(overlap_text)} shared characters:")
            print(f"  \"{preview}…\"")
        else:
            print("  Overlap exists but chunks were split at a clean boundary.")
        print()
        print("  💡 Overlap ensures no sentence is cut in half between chunks.")
        print("     Typical overlap: 10–20% of chunk_size.")

    # ── Different chunk sizes ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("  BONUS: Effect of Chunk Size on Output")
    print("─" * 60)
    for size in [500, 1000, 2000]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 5
        )
        chunks = splitter.split_documents([doc])
        avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
        print(f"    chunk_size={size:>5}  →  {len(chunks):>3} chunks  (avg {avg:>5} chars)")

    # ── Educational notes ───────────────────────────────────────────
    print("\n💡  KEY TAKEAWAYS:")
    print("    1. RecursiveCharacterTextSplitter is the best default — it")
    print("       preserves paragraph and sentence boundaries.")
    print("    2. TokenTextSplitter is essential when you need precise")
    print("       token budgets (for cost control or context windows).")
    print("    3. CharacterTextSplitter is too simplistic for most real use.")
    print("    4. Smaller chunks → more precise retrieval but less context.")
    print("    5. Larger chunks → more context but noisier retrieval.")
    print("    6. Overlap prevents information loss at chunk boundaries.\n")


if __name__ == "__main__":
    main()
