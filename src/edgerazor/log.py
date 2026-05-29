"""
Logging utilities for EdgeRazor.

This module provides a centralized logging system for the EdgeRazor lightweight framework.
It includes specialized loggers for different components like QAT, distillation, etc.
"""

import logging
import random
import sys
import threading
from datetime import datetime
from pathlib import Path

try:
    import pyfiglet
    from colorama import Fore, Style
    from colorama import init as colorama_init
    colorama_init()
    _HAS_ASCII_ART = True
except ImportError:
    _HAS_ASCII_ART = False


class EdgeRazorFormatter(logging.Formatter):
    """Custom formatter for EdgeRazor logs with color support and structured output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_levelname = f"{color}{record.levelname:8s}{self.COLORS['RESET']}"
        component = getattr(record, 'component', 'EdgeRazor')
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        formatted_msg = f"[{timestamp}] [{colored_levelname}] [{component:<10s}] {record.getMessage()}"
        if record.exc_info:
            formatted_msg += f"\n{self.formatException(record.exc_info)}"
        return formatted_msg


# Module-level state to avoid mutable class-variable sharing across instances.
_loggers: dict[str, logging.Logger] = {}
_component_levels: dict[str, int] = {}
_console_handler: logging.StreamHandler | None = None
_file_handler: logging.FileHandler | None = None
_global_level: int = logging.INFO
_initialized: bool = False
_logo_printed: bool = False
_lock: threading.Lock = threading.Lock()

_formatter = EdgeRazorFormatter()


def setup_logging(
    level: str | int = logging.INFO,
    log_file: str | Path | None = None,
    console_output: bool = True,
) -> None:
    """
    Setup the global logging configuration for EdgeRazor.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        console_output: Whether to output logs to console.
    """
    global _initialized, _console_handler, _file_handler, _global_level

    if _initialized:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    _global_level = level

    if console_output:
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setFormatter(_formatter)
        _console_handler.setLevel(logging.DEBUG)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(log_path, encoding='utf-8')
        _file_handler.setFormatter(_formatter)
        _file_handler.setLevel(logging.DEBUG)

    _initialized = True


def set_component_level(component: str, level: str | int) -> None:
    """
    Set the logging level for a specific component.

    This overrides the global level for the named component.
    Call before or after get_logger — existing loggers are updated in-place.

    Args:
        component: Component name (e.g. 'QAT', 'KD').
        level: Logging level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    _component_levels[component] = level

    if component in _loggers:
        _loggers[component].setLevel(level)


def _make_component_method(original_method, component: str):
    """Create a wrapper that merges component info into the log record's extra dict."""
    def component_method(msg, *args, **kwargs):
        extra = dict(kwargs.get('extra', {}))
        extra['component'] = component
        kwargs['extra'] = extra
        return original_method(msg, *args, **kwargs)
    return component_method


def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        component: Component name (e.g. 'QAT', 'KD', 'EdgeRazor').

    Returns:
        Logger instance for the component.
    """
    if not _initialized:
        setup_logging()

    if component not in _loggers:
        with _lock:
            if component not in _loggers:
                logger = logging.getLogger(f"EdgeRazor.{component}")
                level = _component_levels.get(component, _global_level)
                logger.setLevel(level)
                logger.handlers.clear()
                logger.propagate = False

                if _console_handler:
                    logger.addHandler(_console_handler)
                if _file_handler:
                    logger.addHandler(_file_handler)

                logger.debug = _make_component_method(logger.debug, component)
                logger.info = _make_component_method(logger.info, component)
                logger.warning = _make_component_method(logger.warning, component)
                logger.error = _make_component_method(logger.error, component)
                logger.critical = _make_component_method(logger.critical, component)

                _loggers[component] = logger

    return _loggers[component]


def print_logo() -> None:
    """Print the EdgeRazor ASCII art logo once with a randomly selected font."""
    global _logo_printed
    if _logo_printed:
        return
    _logo_printed = True

    if not _HAS_ASCII_ART:
        print("EdgeRazor")
        return

    fonts = ["slant", "cyberlarge", "ansi_shadow"]
    font = random.choice(fonts)
    try:
        result = pyfiglet.figlet_format("EdgeRazor", font=font).rstrip('\n')
        print(Fore.CYAN + result + Style.RESET_ALL)
    except pyfiglet.FontNotFound:
        print(Fore.YELLOW + "EdgeRazor" + Style.RESET_ALL)
