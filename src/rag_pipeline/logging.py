"""src/rag_pipeline/logging.py
Centralized logging configuration for the entire pipeline.
"""
import os
import logging
import sys


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Get a configured logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (default: from LOG_LEVEL env var or INFO)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Use env var for global level control, or default to INFO
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level or env_level)
    
    # Consistent formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent duplicate logs from root logger
    logger.propagate = False
    
    return logger