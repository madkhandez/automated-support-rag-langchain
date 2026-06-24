import re

class OutputFilter:
    """Filters outgoing responses to prevent sensitive data leakage."""
    
    def __init__(self):
        # Patterns to catch potential API keys or secrets leaking
        self.secret_patterns = [
            r"sk-[a-zA-Z0-9]{32,48}", # OpenAI keys
            r"ey[a-zA-Z0-9_=-]+\.ey[a-zA-Z0-9_=-]+\.[a-zA-Z0-9_=-]+", # JWT tokens
        ]
        
    def filter_response(self, answer: str) -> str:
        """Sanitize the outgoing answer."""
        if not answer:
            return answer
            
        filtered_answer = answer
        
        # 1. Remove system prompt leakage
        leakage_phrases = [
            "As an AI language model",
            "Based on the context provided",
            "According to the context",
            "Here is the answer based on the context"
        ]
        
        for phrase in leakage_phrases:
            filtered_answer = filtered_answer.replace(phrase, "")
            
        # 2. Redact potential API keys
        for pattern in self.secret_patterns:
            filtered_answer = re.sub(pattern, "[REDACTED_SECRET]", filtered_answer)
            
        return filtered_answer.strip()
