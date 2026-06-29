"""
LLMFactory — cascading provider fallback with health check.
Priority: Google Gemini → Local Ollama → Anthropic Claude → OpenAI
"""
import os
import logging
from typing import Tuple
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Returns the first healthy LLM provider in the priority chain.
    Falls back gracefully when API keys are missing or providers are down.
    """

    PRIORITY_ORDER = ["google", "local", "anthropic", "openai"]

    # ── Provider loaders ─────────────────────────────────────────
    @staticmethod
    def _load_google() -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.environ["GOOGLE_API_KEY"]  # raises KeyError if missing
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash"),
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

    @staticmethod
    def _load_anthropic() -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic
        key = os.environ["ANTHROPIC_API_KEY"]
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-5"),
            anthropic_api_key=key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        )
        return llm

    @staticmethod
    def _load_openai() -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        key = os.environ["OPENAI_API_KEY"]
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
            openai_api_key=key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        )
        return llm

    # ── Resolver ─────────────────────────────────────────────────
    _LOADERS = {
        "google":    _load_google.__func__,
        "local":     _load_local.__func__,
        "anthropic": _load_anthropic.__func__,
        "openai":    _load_openai.__func__,
    }

    @classmethod
    def get_llm(cls) -> Tuple[BaseChatModel, str]:
        """
        Returns (llm, provider_name).
        Tries each provider in PRIORITY_ORDER; skips on any exception.
        Raises RuntimeError if all providers fail.
        """
        errors = {}
        for name in cls.PRIORITY_ORDER:
            try:
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
            try:
                cls._LOADERS[name]()
                result.append({"provider": name, "status": "available", "priority": cls.PRIORITY_ORDER.index(name) + 1})
            except KeyError:
                result.append({"provider": name, "status": "no_api_key", "priority": cls.PRIORITY_ORDER.index(name) + 1})
            except Exception as e:
                result.append({"provider": name, "status": f"error: {str(e)[:60]}", "priority": cls.PRIORITY_ORDER.index(name) + 1})
        return result
