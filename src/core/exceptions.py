class AetherException(Exception):
    """Base exception for Project Aether."""
    def __init__(self, message: str, status_code: int = 500, *args):
        super().__init__(message, *args)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return f"[AetherException {self.status_code}] {self.message}"

class IngestionException(AetherException):
    """Raised when document ingestion fails."""
    pass

class RetrievalException(AetherException):
    """Raised when document retrieval fails."""
    pass

class CacheException(AetherException):
    """Raised when cache operations fail."""
    pass

class SecurityException(AetherException):
    """Raised when PII masking or security operations fail."""
    pass
