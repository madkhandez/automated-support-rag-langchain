"""
Part 5 — LLM Factory: Centralized LLM and Embeddings Management.

Provides a singleton LLMFactory that creates and caches ChatOpenAI and
OpenAIEmbeddings instances.  Configuration is driven entirely by environment
variables so the same code works in development, staging, and production.

Key features:
  • Singleton pattern — one factory (and one LLM/embeddings object) per process.
  • Health-check method to verify OpenAI connectivity at startup.
  • All secrets loaded via python-dotenv; nothing is ever hardcoded.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Load environment variables ───────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class LLMFactory:
    """Singleton factory for LLM and embedding model instances.

    Usage::

        factory = LLMFactory()            # always returns the same object
        llm     = factory.get_llm()       # ChatOpenAI (cached)
        embeds  = factory.get_embeddings()  # OpenAIEmbeddings (cached)
        ok      = factory.test_connection()  # True / False
    """

    _instance: Optional["LLMFactory"] = None
    _lock: threading.Lock = threading.Lock()

    # Caches keyed by (model_name, temperature) / model_name
    _llm_cache: dict[tuple[str, float], ChatOpenAI] = {}
    _embeddings_cache: dict[str, OpenAIEmbeddings] = {}

    # ── Singleton ────────────────────────────────────────────────────
    def __new__(cls) -> "LLMFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Resolve defaults from env
        self.default_llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.default_embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
        self.api_key: str = os.getenv("OPENAI_API_KEY", "")

        if not self.api_key:
            print("⚠️  OPENAI_API_KEY is not set — LLM calls will fail.")
        else:
            print(f"✅ LLMFactory initialised  |  LLM={self.default_llm_model}  "
                  f"|  Embeddings={self.default_embedding_model}")

    # ── Public API ───────────────────────────────────────────────────
    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> ChatOpenAI:
        """Return a (cached) ChatOpenAI instance.

        Args:
            model: Model name override; falls back to ``LLM_MODEL`` env var.
            temperature: Sampling temperature (0 = deterministic).
        """
        model = model or self.default_llm_model
        cache_key = (model, temperature)

        if cache_key not in self._llm_cache:
            print(f"🔧 Creating ChatOpenAI  model={model}  temp={temperature}")
            self._llm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm_cache[cache_key]

    def get_embeddings(
        self,
        model: Optional[str] = None,
    ) -> OpenAIEmbeddings:
        """Return a (cached) OpenAIEmbeddings instance.

        Args:
            model: Model name override; falls back to ``EMBEDDING_MODEL`` env var.
        """
        model = model or self.default_embedding_model

        if model not in self._embeddings_cache:
            print(f"🔧 Creating OpenAIEmbeddings  model={model}")
            self._embeddings_cache[model] = OpenAIEmbeddings(model=model)
        return self._embeddings_cache[model]

    def test_connection(self) -> bool:
        """Quick smoke-test: embed a tiny string to verify API connectivity."""
        try:
            embeddings = self.get_embeddings()
            result = embeddings.embed_query("health check")
            if result and len(result) > 0:
                print(f"✅ OpenAI connection healthy  (vector dim={len(result)})")
                return True
            print("❌ OpenAI returned an empty embedding vector.")
            return False
        except Exception as exc:
            print(f"❌ OpenAI connection failed: {exc}")
            return False


# ── Standalone entrypoint ────────────────────────────────────────────
def main() -> None:
    """Demonstrate LLMFactory usage."""
    print("=" * 60)
    print("Part 5 · LLM Factory Demo")
    print("=" * 60)

    factory = LLMFactory()

    # 1. Health check
    print("\n── Health check ──")
    healthy = factory.test_connection()
    print(f"   Healthy: {healthy}")

    if not healthy:
        print("\n⚠️  Cannot proceed without a valid OPENAI_API_KEY.  Exiting.")
        return

    # 2. Get an LLM and invoke it
    print("\n── LLM invocation ──")
    llm = factory.get_llm()
    response = llm.invoke("Say 'hello' in three languages.")
    print(f"   Response: {response.content[:200]}")

    # 3. Get embeddings
    print("\n── Embeddings ──")
    embeddings = factory.get_embeddings()
    vector = embeddings.embed_query("production RAG system")
    print(f"   Vector length: {len(vector)}")
    print(f"   First 5 dims:  {vector[:5]}")

    # 4. Singleton verification
    print("\n── Singleton check ──")
    factory2 = LLMFactory()
    print(f"   Same instance: {factory is factory2}")

    print("\n✅ LLM Factory demo complete.")


if __name__ == "__main__":
    main()
