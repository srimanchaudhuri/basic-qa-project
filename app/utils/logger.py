import logging
from functools import lru_cache
import sys

def setup_logger(log_level: str = "INFO") -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

@lru_cache
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

class LoggerMixin:

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)