"""Logging configuration for beam_engine."""

import logging
import os
import sys
from enum import IntEnum
from logging.config import dictConfig
from typing import Optional

_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class LogLevel(IntEnum):
    """Logging levels for beam engine"""
    DEBUG = logging.DEBUG      # 10
    INFO = logging.INFO        # 20
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR      # 40
    CRITICAL = logging.CRITICAL # 50


def _get_default_logging_level() -> str:
    """Get logging level from environment variable or default to INFO"""
    return os.getenv("BEAM_ENGINE_LOGGING_LEVEL", "INFO").upper()


def _should_use_color() -> bool:
    """Determine if colored output should be used"""
    # Check NO_COLOR environment variable (standard)
    if os.getenv("NO_COLOR"):
        return False

    # Check custom color setting
    color_setting = os.getenv("BEAM_ENGINE_LOGGING_COLOR", "auto")
    if color_setting == "0" or color_setting.lower() == "false":
        return False
    if color_setting == "1" or color_setting.lower() == "true":
        return True

    # Auto-detect based on terminal
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
        "colored": {
            "()": ColoredFormatter,
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "colored" if _should_use_color() else "default",
            "level": _get_default_logging_level(),
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "beam_engine": {
            "handlers": ["default"],
            "level": _get_default_logging_level(),
            "propagate": False,
        },
    },
}


def _configure_beam_engine_root_logger() -> None:
    """Configure the root logger for beam_engine"""
    dictConfig(DEFAULT_LOGGING_CONFIG)


def init_logger(name: str) -> logging.Logger:
    """Initialize and return a logger with the given name

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Logger instance configured for beam_engine
    """
    return logging.getLogger(name)


def set_logging_level(level) -> None:
    """Set the logging level for all beam_engine loggers

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as string or LogLevel enum
    """
    # Convert string to uppercase, leave enum/int as-is
    if hasattr(level, 'upper'):
        level = level.upper()
    logger = logging.getLogger("beam_engine")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging() -> None:
    """Disable all beam_engine logging"""
    logger = logging.getLogger("beam_engine")
    logger.disabled = True


def enable_logging() -> None:
    """Enable beam_engine logging"""
    logger = logging.getLogger("beam_engine")
    logger.disabled = False


# Configure the root logger on module import
_configure_beam_engine_root_logger()

# Create the main logger
logger = init_logger(__name__)
