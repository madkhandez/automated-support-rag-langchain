"""
01_document_loaders.py — Document Loading Pipeline for RAG Applications

Demonstrates how to load documents from multiple sources into LangChain
Document objects — the universal currency of the RAG pipeline. Covers:
  • PDF loading via PyPDFLoader (page-level metadata)
  • Plain text loading via TextLoader (with encoding detection)
  • Web scraping via WebBaseLoader (with BeautifulSoup)
  • Bulk directory loading via DirectoryLoader (glob patterns)

Each loader returns List[Document] where every Document carries:
  - page_content: str   (the raw text)
  - metadata: dict      (source file, page number, etc.)

Run:
    python part1_foundation/01_document_loaders.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── LangChain v1 imports ────────────────────────────────────────────
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

# ── Resolve project paths ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")


# ════════════════════════════════════════════════════════════════════
# DocumentLoaderPipeline
# ════════════════════════════════════════════════════════════════════
class DocumentLoaderPipeline:
    """Unified interface for loading documents from PDFs, text files,
    web pages, and entire directories into LangChain Document objects."""

    def __init__(self, docs_dir: str | Path | None = None) -> None:
        self.docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.docs_dir}\n"
                "Make sure the docs/ folder exists relative to the project root."
            )
        print(f"📂  DocumentLoaderPipeline initialised")
        print(f"    Documents directory: {self.docs_dir}")

    # ── PDF Loading ─────────────────────────────────────────────────
    def load_pdf(self, filename: str) -> list[Document]:
        """Load a PDF file using PyPDFLoader (one Document per page).

        Args:
            filename: Name of the PDF file inside self.docs_dir.

        Returns:
            List of Document objects, one per PDF page.
        """
        filepath = self.docs_dir / filename
        if not filepath.exists():
            print(f"⚠️  PDF file not found: {filepath}")
            print("   Tip: Run 'python generate_pdf.py' from the project root to create it.")
            return []

        print(f"\n📄  Loading PDF: {filepath.name}")
        start = time.perf_counter()

        try:
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
        except Exception as exc:
            print(f"❌  Error loading PDF: {exc}")
            return []

        elapsed = time.perf_counter() - start

        print(f"    ✓ Pages loaded     : {len(docs)}")
        print(f"    ✓ Total characters : {sum(len(d.page_content) for d in docs):,}")
        print(f"    ✓ Time elapsed     : {elapsed:.3f}s")

        for i, doc in enumerate(docs):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    Page {i + 1} preview: \"{preview}…\"")
            print(f"    Page {i + 1} metadata: {doc.metadata}")

        return docs

    # ── Text File Loading ───────────────────────────────────────────
    def load_text(self, filename: str, encoding: str = "utf-8") -> list[Document]:
        """Load a plain-text file using TextLoader.

        Args:
            filename: Name of the text file inside self.docs_dir.
            encoding: File encoding (default utf-8).

        Returns:
            List containing a single Document for the entire file.
        """
        filepath = self.docs_dir / filename
        if not filepath.exists():
            print(f"⚠️  Text file not found: {filepath}")
            return []

        print(f"\n📝  Loading text file: {filepath.name}")
        start = time.perf_counter()

        try:
            loader = TextLoader(str(filepath), encoding=encoding)
            docs = loader.load()
        except Exception as exc:
            print(f"❌  Error loading text file: {exc}")
            return []

        elapsed = time.perf_counter() - start

        print(f"    ✓ Documents loaded : {len(docs)}")
        print(f"    ✓ Total characters : {sum(len(d.page_content) for d in docs):,}")
        print(f"    ✓ Time elapsed     : {elapsed:.3f}s")

        for doc in docs:
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"    Preview: \"{preview}…\"")
            print(f"    Metadata: {doc.metadata}")

        return docs

    # ── Web Page Loading ────────────────────────────────────────────
    def load_web(self, urls: list[str]) -> list[Document]:
        """Load web pages using WebBaseLoader with BeautifulSoup.

        Args:
            urls: List of URLs to scrape.

        Returns:
            List of Document objects, one per URL.
        """
        if not urls:
            print("⚠️  No URLs provided for web loading.")
            return []

        print(f"\n🌐  Loading {len(urls)} web page(s)…")
        start = time.perf_counter()

        try:
            loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs={
                    "parse_only": None,  # Load entire page
                },
            )
            docs = loader.load()
        except ImportError:
            print("❌  beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return []
        except Exception as exc:
            print(f"❌  Error loading web pages: {exc}")
            return []

        elapsed = time.perf_counter() - start

        print(f"    ✓ Pages loaded     : {len(docs)}")
        print(f"    ✓ Total characters : {sum(len(d.page_content) for d in docs):,}")
        print(f"    ✓ Time elapsed     : {elapsed:.3f}s")

        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:100].replace("\n", " ").strip()
            print(f"    Source : {source}")
            print(f"    Preview: \"{preview}…\"")

        return docs

    # ── Directory Loading ───────────────────────────────────────────
    def load_directory(
        self,
        glob_pattern: str = "**/*.*",
        show_progress: bool = True,
    ) -> list[Document]:
        """Load all supported files from a directory using DirectoryLoader.

        Supports .txt, .md, and .pdf files by default. Each file type is
        routed to the appropriate loader via the loader_cls parameter.

        Args:
            glob_pattern: Glob pattern for file discovery.
            show_progress: Whether to display a progress indicator.

        Returns:
            List of Document objects from all loaded files.
        """
        print(f"\n📁  Loading directory: {self.docs_dir}")
        print(f"    Pattern: {glob_pattern}")
        start = time.perf_counter()

        all_docs: list[Document] = []

        # Load each file type separately for correct parsing
        loaders_config: list[tuple[str, type, dict]] = [
            ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
            ("**/*.md", TextLoader, {"encoding": "utf-8"}),
            ("**/*.pdf", PyPDFLoader, {}),
        ]

        for pattern, loader_cls, loader_kwargs in loaders_config:
            try:
                dir_loader = DirectoryLoader(
                    str(self.docs_dir),
                    glob=pattern,
                    loader_cls=loader_cls,
                    loader_kwargs=loader_kwargs,
                    show_progress=show_progress,
                    use_multithreading=False,
                    silent_errors=True,
                )
                docs = dir_loader.load()
                if docs:
                    print(f"    ✓ {pattern}: loaded {len(docs)} document(s)")
                    all_docs.extend(docs)
            except Exception as exc:
                print(f"    ⚠️  Error with pattern '{pattern}': {exc}")

        elapsed = time.perf_counter() - start

        print(f"\n    ── Directory Summary ──────────────────────")
        print(f"    Total documents    : {len(all_docs)}")
        print(f"    Total characters   : {sum(len(d.page_content) for d in all_docs):,}")
        print(f"    Time elapsed       : {elapsed:.3f}s")

        return all_docs

    # ── Summary Report ──────────────────────────────────────────────
    @staticmethod
    def print_summary(docs: list[Document], label: str = "All Documents") -> None:
        """Print a detailed summary of loaded documents.

        Args:
            docs: List of Document objects to summarise.
            label: Label for the summary header.
        """
        if not docs:
            print(f"\n📊  {label}: No documents to summarise.")
            return

        total_chars = sum(len(d.page_content) for d in docs)
        sources = set()
        source_types: dict[str, int] = {}

        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            sources.add(src)
            # Determine source type by extension
            ext = Path(src).suffix.lower() if not src.startswith("http") else ".web"
            source_types[ext] = source_types.get(ext, 0) + 1

        print(f"\n{'═' * 60}")
        print(f"📊  SUMMARY: {label}")
        print(f"{'═' * 60}")
        print(f"  Total documents  : {len(docs)}")
        print(f"  Total characters : {total_chars:,}")
        print(f"  Unique sources   : {len(sources)}")
        print(f"  Avg doc length   : {total_chars // len(docs):,} chars")
        print()
        print(f"  Source types:")
        for ext, count in sorted(source_types.items()):
            print(f"    {ext or '(none)':>8} : {count} document(s)")
        print()
        print(f"  Source files:")
        for src in sorted(sources):
            print(f"    • {src}")
        print(f"{'═' * 60}")

    # ── Unsupported format check ────────────────────────────────────
    @staticmethod
    def check_supported(filename: str) -> bool:
        """Check whether a file extension is supported.

        Args:
            filename: File name or path to check.

        Returns:
            True if the format is supported, False otherwise.
        """
        supported = {".txt", ".md", ".pdf", ".html", ".csv", ".json"}
        ext = Path(filename).suffix.lower()
        if ext not in supported:
            print(f"⚠️  Unsupported file format: '{ext}' — "
                  f"supported formats: {', '.join(sorted(supported))}")
            return False
        return True


# ════════════════════════════════════════════════════════════════════
# Main demonstration
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run all document loaders and display results."""
    print("=" * 60)
    print("  Part 1.1 — Document Loaders for RAG")
    print("=" * 60)

    pipeline = DocumentLoaderPipeline()
    all_docs: list[Document] = []

    # ── 1. Load PDF ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 1: PDF Loading with PyPDFLoader")
    print("─" * 60)
    pdf_docs = pipeline.load_pdf("langchain_demo.pdf")
    all_docs.extend(pdf_docs)

    # ── 2. Load Text ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 2: Text File Loading with TextLoader")
    print("─" * 60)
    txt_docs = pipeline.load_text("company_policy.txt")
    all_docs.extend(txt_docs)

    # ── 3. Load Markdown (also via TextLoader) ──────────────────────
    print("\n" + "─" * 60)
    print("  STEP 3: Markdown File Loading with TextLoader")
    print("─" * 60)
    md_docs = pipeline.load_text("tech_docs.md")
    all_docs.extend(md_docs)

    # ── 4. Web Loading ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 4: Web Page Loading with WebBaseLoader")
    print("─" * 60)
    web_urls = [
        "https://docs.smith.langchain.com/overview",
    ]
    print("  ℹ️  Web loading requires internet access.")
    print("     Attempting to load LangSmith docs page…")
    web_docs = pipeline.load_web(web_urls)
    all_docs.extend(web_docs)

    # ── 5. Directory Loading ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 5: Bulk Directory Loading with DirectoryLoader")
    print("─" * 60)
    dir_docs = pipeline.load_directory()
    # Note: dir_docs already contains the same files — shown for demo only.
    # We don't add them to all_docs to avoid duplicates.
    print(f"\n  ℹ️  Directory loader found {len(dir_docs)} documents total.")
    print("     (Not added to all_docs to avoid duplicates.)")

    # ── 6. Unsupported format check ─────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 6: Format Validation")
    print("─" * 60)
    test_files = ["report.pdf", "data.csv", "photo.jpg", "archive.zip"]
    for fname in test_files:
        pipeline.check_supported(fname)

    # ── Final summary ───────────────────────────────────────────────
    pipeline.print_summary(all_docs, label="All Loaded Documents")

    # ── Educational note ────────────────────────────────────────────
    print("\n💡  KEY TAKEAWAYS:")
    print("    1. Every loader returns List[Document] — a universal format.")
    print("    2. Metadata (source, page number) is automatically attached.")
    print("    3. PDFs produce one Document per page; text files produce one total.")
    print("    4. DirectoryLoader lets you ingest entire folders at once.")
    print("    5. WebBaseLoader uses BeautifulSoup for HTML parsing.")
    print("    6. Always check for supported formats before loading.\n")


if __name__ == "__main__":
    main()
