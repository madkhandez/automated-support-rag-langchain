import time
from collections import defaultdict
from typing import Dict, List

class RateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Store timestamps of requests per user: {user_id: [timestamp1, timestamp2, ...]}
        self.requests: Dict[str, List[float]] = defaultdict(list)
        
    def _cleanup(self, user_id: str, current_time: float):
        """Remove timestamps older than the window."""
        window_start = current_time - self.window_seconds
        self.requests[user_id] = [ts for ts in self.requests[user_id] if ts > window_start]
        
    def check(self, user_id: str) -> bool:
        """
        Check if user is allowed to make a request.
        Returns True if allowed, False if rate limited.
        """
        current_time = time.time()
        self._cleanup(user_id, current_time)
        
        if len(self.requests[user_id]) >= self.max_requests:
            return False
            
        # Record the request
        self.requests[user_id].append(current_time)
        return True
        
    def get_remaining(self, user_id: str) -> int:
        """Get number of remaining requests allowed in current window."""
        current_time = time.time()
        self._cleanup(user_id, current_time)
        return max(0, self.max_requests - len(self.requests[user_id]))
