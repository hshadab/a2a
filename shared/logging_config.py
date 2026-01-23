"""
Logging configuration for the Threat Intelligence Network
"""
import logging
import sys
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for an agent or module.

    Args:
        name: Logger name (usually __name__ or agent name)
        level: Logging level (default INFO)
        log_format: Optional custom format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


# Pre-configured loggers for each component
scout_logger = setup_logging("scout")
policy_logger = setup_logging("policy")
analyst_logger = setup_logging("analyst")
database_logger = setup_logging("database")
prover_logger = setup_logging("prover")
x402_logger = setup_logging("x402")
a2a_logger = setup_logging("a2a")
