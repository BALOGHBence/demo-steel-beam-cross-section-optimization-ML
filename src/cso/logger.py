import logging
from functools import lru_cache
from rich.logging import RichHandler


@lru_cache(maxsize=1, typed=False)
def get_logger() -> logging.Logger:
    """Returns a logger instance for cross_section_optimization."""
    logger = logging.getLogger("cross_section_optimization")

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,   # optional: hide file path in output
        )

        # RichHandler uses only %(message)s by default
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def set_log_level(level: str | int) -> None:
    """Sets log level for the project logger."""
    logger = get_logger()
    # Normalize level to int if it's a string
    if isinstance(level, str):
        level_int = logging._nameToLevel.get(level.upper(), logging.INFO)
    else:
        level_int = level
    logger.setLevel(level_int)
    logger.info(f"Log level set to {logging.getLevelName(level_int)}")