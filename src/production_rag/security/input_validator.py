import re
from typing import Dict, Any
from .rate_limiter import RateLimiter

class InputSecurityLayer:
    """Validates and sanitizes all input to the RAG system."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        
        # Patterns indicative of prompt injection attempts
        self.injection_patterns = [
            r"ignore previous",
            r"system:",
            r"you are now",
            r"jailbreak",
            r"forget your instructions",
            r"DAN mode activated",
            r"override your programming",
            r"disregard all previous",
            r"new instructions:",
            r"sudo mode"
        ]
        
        # PII Detection patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
        }

    def validate_question(self, text: str) -> Dict[str, Any]:
        """Validate a user question."""
        if not text or len(text.strip()) == 0:
            return {"is_valid": False, "reason": "Question cannot be empty"}
            
        if len(text) > 2000:
            return {"is_valid": False, "reason": "Question exceeds maximum length of 2000 characters"}
            
        # Check for prompt injection
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return {"is_valid": False, "reason": "Invalid prompt format detected"}
                
        # Detect and warn about PII (in a real app, we might block or redact)
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                print(f"⚠️ Security Warning: Detected potential {pii_type} in input.")
                # We could implement redaction here: text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
                
        return {"is_valid": True, "reason": "Valid"}

    def validate_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate an uploaded document."""
        # Check size (10MB max)
        max_size = 10 * 1024 * 1024
        if len(file_content) > max_size:
            return {"is_valid": False, "reason": "File size exceeds 10MB limit"}
            
        # Check extension
        allowed_extensions = {".pdf", ".txt", ".md"}
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in allowed_extensions:
            return {"is_valid": False, "reason": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"}
            
        return {"is_valid": True, "reason": "Valid"}

    def rate_limit_check(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        return self.rate_limiter.check(user_id)
