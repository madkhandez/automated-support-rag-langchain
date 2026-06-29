"""
LLM Factory: Centralized LLM and Embeddings Management.

Provides a singleton LLMFactory that creates and caches LLM and embedding
instances.  Configuration is driven entirely by environment variables so the
same code works in development, staging, and production.

Key features:
  • Singleton pattern — one factory (and one LLM/embeddings object) per process.
  • Multi-provider LLM fallback chain (OpenAI → Anthropic → Google) via
    LangChain's ``.with_fallbacks()`` for maximum uptime.
  • Local HuggingFace embeddings (BAAI/bge-m3) — zero API cost, excellent
    multilingual / Indonesian support, 1024-dim vectors.
  • Health-check method to verify embeddings at startup.
  • All secrets loaded via python-dotenv; nothing is ever hardcoded.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ── Load environment variables ───────────────────────────────────────
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)


class LLMFactory:
    """Singleton factory for LLM and embedding model instances.

    Usage::

        factory = LLMFactory()            # always returns the same object
        llm     = factory.get_llm()       # LLM with fallback chain (cached)
        embeds  = factory.get_embeddings()  # HuggingFaceEmbeddings (cached)
        ok      = factory.test_connection()  # True / False
    """

    _instance: Optional["LLMFactory"] = None
    _lock: threading.Lock = threading.Lock()

    # Caches keyed by (model_name, temperature) / model_name
    _llm_cache: dict[tuple[str, float], Any] = {}
    _embeddings_cache: dict[str, Any] = {}

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
        self.default_llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.default_embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))

        # API keys (optional — fallback chain degrades gracefully)
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

        if not self.openai_api_key:
            print("⚠️  OPENAI_API_KEY is not set — primary LLM will fail, "
                  "fallbacks may still work.")
        else:
            print(f"✅ LLMFactory initialised  |  LLM={self.default_llm_model}  "
                  f"|  Embeddings={self.default_embedding_model}")

    # ── Private helpers ──────────────────────────────────────────────
    @staticmethod
    def _is_real_key(key: str) -> bool:
        """Check if an API key looks real (not empty or a placeholder)."""
        if not key:
            return False
        placeholders = {"your_", "sk-your", "put_", "insert_", "change_", "replace_"}
        return not any(key.lower().startswith(p) for p in placeholders)

    def _build_fallback_chain(self, temperature: float) -> Any:
        """Build a primary LLM with available fallback providers.

        Only includes providers whose API keys are set to real values.
        Uses LangChain's ``.with_fallbacks()`` so that if the primary model
        fails, the request is transparently retried with the next provider.

        Returns:
            A ``RunnableWithFallbacks`` if fallbacks are available,
            or a plain ``ChatOpenAI`` if no fallbacks are configured.
        """
        # ── Primary: OpenAI ──────────────────────────────────────────
        primary = ChatOpenAI(
            model=self.default_llm_model,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        print(f"   ✅ Primary   → ChatOpenAI({self.default_llm_model})")

        # ── Collect available fallbacks ──────────────────────────────
        fallbacks = []

        # Fallback 1: Anthropic
        if self._is_real_key(self.anthropic_api_key):
            from langchain_anthropic import ChatAnthropic
            fallbacks.append(ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=temperature,
                max_tokens=self.max_tokens,
            ))
            print("   ✅ Fallback1 → ChatAnthropic(claude-3-haiku-20240307)")
        else:
            print("   ⏭️  Fallback1 → Anthropic SKIPPED (ANTHROPIC_API_KEY not set)")

        # Fallback 2: Google
        if self._is_real_key(self.google_api_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            fallbacks.append(ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                max_tokens=self.max_tokens,
            ))
            print("   ✅ Fallback2 → ChatGoogleGenerativeAI(gemini-1.5-flash)")
        else:
            print("   ⏭️  Fallback2 → Google SKIPPED (GOOGLE_API_KEY not set)")

        # Build the chain
        if fallbacks:
            return primary.with_fallbacks(fallbacks)

        print("   ⚠️  No fallback providers configured — using OpenAI only")
        return primary

    # ── Public API ───────────────────────────────────────────────────
    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Any:
        """Return a (cached) LLM with fallback chain.

        The returned object is a ``RunnableWithFallbacks``:
          Primary:   ChatOpenAI (gpt-4o-mini)
          Fallback1: ChatAnthropic (claude-3-haiku-20240307)
          Fallback2: ChatGoogleGenerativeAI (gemini-1.5-flash)

        Args:
            model: Ignored in fallback mode (reserved for future use).
            temperature: Sampling temperature (0 = deterministic).
        """
        cache_key = (self.default_llm_model, temperature)

        if cache_key not in self._llm_cache:
            print(f"🔧 Building LLM fallback chain  temp={temperature}")
            self._llm_cache[cache_key] = self._build_fallback_chain(temperature)
        return self._llm_cache[cache_key]

    def get_embeddings(
        self,
        model: Optional[str] = None,
    ) -> Any:
        """Return a (cached) HuggingFaceEmbeddings instance.

        Uses ``BAAI/bge-m3`` by default — a high-quality multilingual embedding
        model that runs entirely locally (no API calls, no cost).

        Args:
            model: Model name override; falls back to ``EMBEDDING_MODEL`` env var.
        """
        from langchain_huggingface import HuggingFaceEmbeddings

        model = model or self.default_embedding_model

        if model not in self._embeddings_cache:
            print(f"🔧 Loading HuggingFaceEmbeddings  model={model}")
            self._embeddings_cache[model] = HuggingFaceEmbeddings(
                model_name=model,
            )
            print(f"   ✅ Embeddings loaded (local, zero API cost)")
        return self._embeddings_cache[model]

    def test_connection(self) -> bool:
        """Quick smoke-test: embed a tiny string to verify embeddings work."""
        try:
            embeddings = self.get_embeddings()
            result = embeddings.embed_query("health check")
            if result and len(result) > 0:
                print(f"✅ Embeddings healthy  (vector dim={len(result)}, "
                      f"model={self.default_embedding_model})")
                return True
            print("❌ Embeddings returned an empty vector.")
            return False
        except Exception as exc:
            print(f"❌ Embeddings test failed: {exc}")
            return False


# ── Standalone entrypoint ────────────────────────────────────────────
def main() -> None:
    """Demonstrate LLMFactory usage."""
    print("=" * 60)
    print("LLM Factory Demo — Fallback Chain + Local Embeddings")
    print("=" * 60)

    factory = LLMFactory()

    # 1. Health check (embeddings)
    print("\n── Embeddings health check ──")
    healthy = factory.test_connection()
    print(f"   Healthy: {healthy}")

    if not healthy:
        print("\n⚠️  Embeddings failed to load.  Exiting.")
        return

    # 2. Get embeddings
    print("\n── Embeddings ──")
    embeddings = factory.get_embeddings()
    vector = embeddings.embed_query("production RAG system")
    print(f"   Vector length: {len(vector)}")
    print(f"   First 5 dims:  {vector[:5]}")

    # 3. Get the LLM (fallback chain) and invoke it
    print("\n── LLM invocation (fallback chain) ──")
    llm = factory.get_llm()
    try:
        response = llm.invoke("Say 'hello' in three languages.")
        print(f"   Response: {response.content[:200]}")
    except Exception as exc:
        print(f"   ⚠️  All LLM providers failed: {exc}")

    # 4. Singleton verification
    print("\n── Singleton check ──")
    factory2 = LLMFactory()
    print(f"   Same instance: {factory is factory2}")

    print("\n✅ LLM Factory demo complete.")


if __name__ == "__main__":
    main()
