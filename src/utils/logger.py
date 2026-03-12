import logging
import json
from datetime import datetime
from src.utils.config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "service": "project_aether",
            "environment": getattr(settings, "environment", "development")
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logger() -> logging.Logger:
    """Configures and returns the root logger with JSON formatting."""
    logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger

# Initialize logger globally
logger = setup_logger()
