import sys
import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False

log_format = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not logger.handlers:
    logger.addHandler(console_handler)

def time_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start
            logger.info(f"Function '{func.__name__}' executed in {duration:.4f}s")
    return wrapper
