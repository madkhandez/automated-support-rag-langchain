"""
LLMFactory — cascading provider fallback with health check.
Priority: Google Gemini → Local Ollama → Anthropic Claude → OpenAI

Google Gemini has an internal model cascade: when a model hits its
rate limit the factory automatically tries the next model in the list
before falling back to the next provider.
"""
import os
import logging
from typing import List, Tuple
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ── Google model cascade (ordered by performance, best first) ────
# Flash models first (higher quality), then Flash Lite (lower cost).
_DEFAULT_GOOGLE_CASCADE: List[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


def _get_google_model_cascade() -> List[str]:
    """Build the ordered list of Google models to try.

    Priority:
    1. GOOGLE_LLM_MODEL env var (if set and not already in cascade)
    2. GOOGLE_MODEL_CASCADE env var (comma-separated, overrides default)
    3. _DEFAULT_GOOGLE_CASCADE
    """
    env_cascade = os.getenv("GOOGLE_MODEL_CASCADE", "")
    if env_cascade.strip():
        cascade = [m.strip() for m in env_cascade.split(",") if m.strip()]
    else:
        cascade = list(_DEFAULT_GOOGLE_CASCADE)

    # If the user explicitly set GOOGLE_LLM_MODEL, prioritise it first.
    preferred = os.getenv("GOOGLE_LLM_MODEL", "").strip()
    if preferred and preferred not in cascade:
        cascade.insert(0, preferred)
    elif preferred and preferred in cascade:
        # Move it to the front so the user's preference always wins.
        cascade.remove(preferred)
        cascade.insert(0, preferred)

    return cascade


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a rate-limit / quota error.

    Covers Google's common error shapes:
    • google.api_core.exceptions.ResourceExhausted (gRPC)
    • HTTP 429 responses surfaced by langchain-google-genai
    • Generic "quota" / "rate limit" messages from the SDK
    """
    exc_type = type(exc).__name__

    # Direct class check (google.api_core.exceptions.ResourceExhausted)
    if exc_type in ("ResourceExhausted", "TooManyRequests"):
        return True

    msg = str(exc).lower()
    rate_limit_signals = [
        "429",
        "resource exhausted",
        "resourceexhausted",
        "rate limit",
        "rate_limit",
        "quota",
        "too many requests",
    ]
    return any(signal in msg for signal in rate_limit_signals)


class LLMFactory:
    """
    Returns the first healthy LLM provider in the priority chain.
    Falls back gracefully when API keys are missing or providers are down.

    Google provider has an internal model cascade: when a model hits its
    rate limit, the next model in GOOGLE_MODEL_CASCADE is tried before
    moving to the next provider.
    """

    PRIORITY_ORDER = ["google", "local", "anthropic", "openai"]

    # ── Provider loaders ─────────────────────────────────────────

    @staticmethod
    def _load_google(model: str) -> BaseChatModel:
        """Load a single Google Gemini model.  Caller handles cascade."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.environ["GOOGLE_API_KEY"]  # raises KeyError if missing
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            convert_system_message_to_human=True,
        )
        return llm

    @staticmethod
    def _load_local() -> BaseChatModel:
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )
        return llm

    # @staticmethod
    # def _load_anthropic() -> BaseChatModel:
    #     from langchain_anthropic import ChatAnthropic
    #     key = os.environ["ANTHROPIC_API_KEY"]
    #     llm = ChatAnthropic(
    #         model=os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-5"),
    #         anthropic_api_key=key,
    #         temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    #         max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
    #     )
    #     return llm

    # @staticmethod
    # def _load_openai() -> BaseChatModel:
    #     from langchain_openai import ChatOpenAI
    #     key = os.environ["OPENAI_API_KEY"]
    #     llm = ChatOpenAI(
    #         model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
    #         openai_api_key=key,
    #         temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    #         max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
    #     )
    #     return llm

    # ── Resolver ─────────────────────────────────────────────────
    _LOADERS = {
        "local":     _load_local.__func__,
        # "anthropic": _load_anthropic.__func__,
        # "openai":    _load_openai.__func__,
    }

    @classmethod
    def _try_google_cascade(cls) -> Tuple[BaseChatModel, str]:
        """Try each Google model in the cascade; raise on non-rate-limit errors.

        Returns:
            (llm, "google/<model_name>") on success.

        Raises:
            KeyError   — GOOGLE_API_KEY not set (caller should skip provider).
            RuntimeError — All models exhausted due to rate limits.
            Exception  — Non-rate-limit error (caller should skip provider).
        """
        cascade = _get_google_model_cascade()
        rate_limit_errors: dict[str, str] = {}

        for model in cascade:
            try:
                llm = cls._load_google(model)
                # Smoke test — cheap single-token call
                llm.invoke("hi")
                logger.info(
                    f"[LLMFactory] Using Google model: {model}"
                )
                return llm, f"google/{model}"
            except KeyError:
                # API key missing — no point trying other models.
                raise
            except Exception as e:
                if _is_rate_limit_error(e):
                    rate_limit_errors[model] = str(e)
                    logger.warning(
                        f"[LLMFactory] Google model '{model}' rate-limited, "
                        f"trying next model…"
                    )
                    continue
                else:
                    # Non-rate-limit error (auth, invalid model, etc.)
                    # Re-raise so the outer loop can skip the google provider.
                    raise

        # All models exhausted due to rate limits.
        raise RuntimeError(
            f"All Google models rate-limited. Tried: {list(rate_limit_errors.keys())}"
        )

    @classmethod
    def get_llm(cls) -> Tuple[BaseChatModel, str]:
        """
        Returns (llm, provider_name).
        Tries each provider in PRIORITY_ORDER; skips on any exception.

        For the Google provider, cascades through all configured models
        before falling back to the next provider.

        Raises RuntimeError if all providers fail.
        """
        errors = {}
        for name in cls.PRIORITY_ORDER:
            try:
                if name == "google":
                    llm, provider_label = cls._try_google_cascade()
                    logger.info(f"[LLMFactory] Using provider: {provider_label}")
                    return llm, provider_label
                else:
                    llm = cls._LOADERS[name]()
                    # Smoke test — cheap single-token call
                    llm.invoke("hi")
                    logger.info(f"[LLMFactory] Using provider: {name}")
                    return llm, name
            except KeyError:
                errors[name] = "API key not configured"
                logger.debug(f"[LLMFactory] {name}: API key missing, skipping")
            except Exception as e:
                errors[name] = str(e)
                logger.warning(f"[LLMFactory] {name} unavailable: {e}")

        raise RuntimeError(
            f"All LLM providers failed. Details: {errors}\n"
            "Set at least one of: GOOGLE_API_KEY, OLLAMA_BASE_URL, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY in your .env file."
        )

    @classmethod
    def get_available_providers(cls) -> list[dict]:
        """Returns status of all providers without throwing."""
        result = []
        for name in cls.PRIORITY_ORDER:
            priority = cls.PRIORITY_ORDER.index(name) + 1
            if name == "google":
                cascade = _get_google_model_cascade()
                try:
                    _ = os.environ["GOOGLE_API_KEY"]
                    result.append({
                        "provider": "google",
                        "status": "available",
                        "priority": priority,
                        "models": cascade,
                    })
                except KeyError:
                    result.append({
                        "provider": "google",
                        "status": "no_api_key",
                        "priority": priority,
                        "models": cascade,
                    })
            else:
                try:
                    cls._LOADERS[name]()
                    result.append({"provider": name, "status": "available", "priority": priority})
                except KeyError:
                    result.append({"provider": name, "status": "no_api_key", "priority": priority})
                except Exception as e:
                    result.append({"provider": name, "status": f"error: {str(e)[:60]}", "priority": priority})
        return result
