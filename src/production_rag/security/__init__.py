"""
Production security module for RAG applications.

Provides input validation, output filtering, rate limiting, and security management
to protect against prompt injection, PII leakage, and abuse.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    reason: Optional[str] = None
    sanitized_input: Optional[str] = None


class InputValidator:
    """Validates and sanitizes user inputs for security threats.
    
    Detects prompt injection attempts, PII in queries, oversized inputs,
    and unsupported file types.
    """

    # Known prompt injection patterns
    INJECTION_PATTERNS: list[str] = [
        r"ignore\s+previous\s+instructions",
        r"system:\s*you\s+are\s+now",
        r"forget\s+your\s+instructions",
        r"jailbreak\s+mode",
        r"dan\s+mode\s+activated",
        r"you\s+are\s+now\s+dan",
        r"override\s+your\s+programming",
        r"disregard\s+all\s+previous",
        r"new\s+instructions:",
        r"sudo\s+mode",
    ]

    # PII detection patterns
    PII_PATTERNS: dict[str, str] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    ALLOWED_FILE_EXTENSIONS: set[str] = {
        ".pdf", ".txt", ".md", ".csv", ".json", ".docx", ".html", ".htm",
    }

    MAX_INPUT_LENGTH: int = 2000

    def __init__(self, max_input_length: int = 2000) -> None:
        """Initialize with configurable max input length."""
        self.MAX_INPUT_LENGTH = max_input_length

    def detect_prompt_injection(self, text: str) -> bool:
        """Check if text contains prompt injection patterns.
        
        Args:
            text: The input text to check.
            
        Returns:
            True if injection is detected, False otherwise.
        """
        text_lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def detect_pii(self, text: str) -> dict[str, bool]:
        """Check if text contains PII patterns.
        
        Args:
            text: The input text to check.
            
        Returns:
            Dictionary mapping PII type to whether it was detected.
        """
        results: dict[str, bool] = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            results[pii_type] = bool(re.search(pattern, text))
        return results

    def validate_input_length(self, text: str) -> ValidationResult:
        """Check if input is within acceptable length.
        
        Args:
            text: The input text to validate.
            
        Returns:
            ValidationResult indicating if input length is acceptable.
        """
        if len(text) > self.MAX_INPUT_LENGTH:
            return ValidationResult(
                is_valid=False,
                reason=f"Input exceeds maximum length of {self.MAX_INPUT_LENGTH} characters "
                       f"(got {len(text)})",
            )
        return ValidationResult(is_valid=True, sanitized_input=text)

    def validate_file_type(self, filename: str) -> ValidationResult:
        """Check if file type is allowed.
        
        Args:
            filename: The filename to validate.
            
        Returns:
            ValidationResult indicating if the file type is allowed.
        """
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_FILE_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                reason=f"File type '{ext}' is not supported. "
                       f"Allowed types: {', '.join(sorted(self.ALLOWED_FILE_EXTENSIONS))}",
            )
        return ValidationResult(is_valid=True, sanitized_input=filename)

    def validate_question(self, question: str) -> ValidationResult:
        """Full validation pipeline for a user question.
        
        Checks injection, PII, and length in sequence.
        
        Args:
            question: The user's question to validate.
            
        Returns:
            ValidationResult from the first failing check, or valid result.
        """
        # Check prompt injection
        if self.detect_prompt_injection(question):
            return ValidationResult(
                is_valid=False,
                reason="Potential prompt injection detected",
            )

        # Check PII
        pii_results = self.detect_pii(question)
        if any(pii_results.values()):
            detected_types = [t for t, found in pii_results.items() if found]
            return ValidationResult(
                is_valid=False,
                reason=f"PII detected in input: {', '.join(detected_types)}",
            )

        # Check length
        length_result = self.validate_input_length(question)
        if not length_result.is_valid:
            return length_result

        return ValidationResult(is_valid=True, sanitized_input=question.strip())


class OutputFilter:
    """Filters sensitive information from LLM outputs."""

    # Patterns to redact from output
    SENSITIVE_PATTERNS: dict[str, str] = {
        "api_key": r"(?:sk-|api[_-]?key[=:\s]+)[a-zA-Z0-9\-_]{20,}",
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "password": r"(?:password|passwd|pwd)[=:\s]+\S+",
    }

    def filter_output(self, text: str) -> str:
        """Remove sensitive patterns from output text.
        
        Args:
            text: The output text to filter.
            
        Returns:
            Filtered text with sensitive patterns replaced by [REDACTED].
        """
        filtered = text
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            filtered = re.sub(pattern, "[REDACTED]", filtered, flags=re.IGNORECASE)
        return filtered


class RateLimiter:
    """Token-bucket rate limiter for API requests.
    
    Tracks request timestamps per client and enforces limits
    within a rolling time window.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the time window.
            window_seconds: Size of the rolling time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is allowed to make a request.
        
        Args:
            client_id: Unique identifier for the client.
            
        Returns:
            True if the request is allowed, False if rate limited.
        """
        now = time.time()
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Remove expired timestamps
        self._requests[client_id] = [
            ts for ts in self._requests[client_id]
            if now - ts < self.window_seconds
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset rate limiter state.
        
        Args:
            client_id: If provided, reset only this client. Otherwise reset all.
        """
        if client_id:
            self._requests.pop(client_id, None)
        else:
            self._requests.clear()


class SecurityManager:
    """Orchestrates all security components for the RAG pipeline."""

    def __init__(
        self,
        max_input_length: int = 2000,
        max_requests: int = 20,
        window_seconds: int = 60,
    ) -> None:
        self.input_validator = InputValidator(max_input_length=max_input_length)
        self.output_filter = OutputFilter()
        self.rate_limiter = RateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def validate_and_process(
        self, question: str, client_id: str = "default"
    ) -> ValidationResult:
        """Full security check: rate limit + input validation.
        
        Args:
            question: The user's question.
            client_id: Client identifier for rate limiting.
            
        Returns:
            ValidationResult from the first failing check, or valid result.
        """
        if not self.rate_limiter.is_allowed(client_id):
            return ValidationResult(
                is_valid=False,
                reason="Rate limit exceeded. Please try again later.",
            )
        return self.input_validator.validate_question(question)

    def filter_response(self, response: str) -> str:
        """Filter sensitive content from response."""
        return self.output_filter.filter_output(response)
