class ProjectAetherException(Exception):
    """Base exception for Project Aether."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class IngestionException(ProjectAetherException):
    """Exception raised during document ingestion."""
    pass

class RetrievalException(ProjectAetherException):
    """Exception raised during query retrieval."""
    pass
