"""Unit tests for EdgeRazor logging module."""

import logging

import pytest

from edgerazor.log import (
    EdgeRazorFormatter,
    get_logger,
    print_logo,
    set_component_level,
    setup_logging,
)


class TestEdgeRazorFormatter:
    def test_format_debug_message(self):
        formatter = EdgeRazorFormatter()
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        record.component = "TestComp"
        output = formatter.format(record)
        assert "test message" in output
        assert "DEBUG" in output
        assert "TestComp" in output

    def test_format_info_message(self):
        formatter = EdgeRazorFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="info message", args=(), exc_info=None,
        )
        record.component = "QAT"
        output = formatter.format(record)
        assert "info message" in output
        assert "INFO" in output
        assert "QAT" in output

    def test_format_error_message(self):
        formatter = EdgeRazorFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error occurred", args=(), exc_info=None,
        )
        record.component = "EdgeRazor"
        output = formatter.format(record)
        assert "error occurred" in output
        assert "ERROR" in output

    def test_default_component(self):
        formatter = EdgeRazorFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "EdgeRazor" in output

    def test_format_with_exception(self):
        import sys
        formatter = EdgeRazorFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error", args=(), exc_info=sys.exc_info(),
            )
        record.component = "KD"
        output = formatter.format(record)
        assert "ValueError" in output
        assert "test error" in output


class TestSetupLogging:
    def test_setup_logging_default(self):
        from edgerazor.log import _initialized
        # Reset state for test
        import edgerazor.log as log_mod
        log_mod._initialized = False
        log_mod._loggers.clear()

        setup_logging(level=logging.WARNING)
        assert log_mod._global_level == logging.WARNING
        assert log_mod._initialized is True

    def test_setup_logging_idempotent(self):
        import edgerazor.log as log_mod
        log_mod._initialized = True
        log_mod._global_level = logging.INFO

        setup_logging(level=logging.DEBUG)
        # Should not change since already initialized
        assert log_mod._global_level == logging.INFO

    def test_setup_logging_with_level_string(self):
        import edgerazor.log as log_mod
        log_mod._initialized = False
        log_mod._loggers.clear()

        setup_logging(level="ERROR")
        assert log_mod._global_level == logging.ERROR
        log_mod._initialized = False


class TestGetLogger:
    def test_get_logger_creates_logger(self):
        import edgerazor.log as log_mod
        log_mod._initialized = False
        log_mod._loggers.clear()
        setup_logging(level=logging.CRITICAL)

        logger = get_logger("TestComponent")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_same_instance(self):
        logger1 = get_logger("SameComponent")
        logger2 = get_logger("SameComponent")
        assert logger1 is logger2

    def test_get_logger_different_components(self):
        logger_a = get_logger("ComponentA")
        logger_b = get_logger("ComponentB")
        assert logger_a is not logger_b


class TestSetComponentLevel:
    def test_set_component_level(self):
        import edgerazor.log as log_mod
        log_mod._initialized = False
        log_mod._loggers.clear()
        log_mod._component_levels.clear()
        setup_logging(level=logging.CRITICAL)

        set_component_level("QAT", "DEBUG")
        assert log_mod._component_levels.get("QAT") == logging.DEBUG

        logger = get_logger("QAT")
        assert logger.level == logging.DEBUG


class TestPrintLogo:
    def test_print_logo_runs_once(self, capsys):
        import edgerazor.log as log_mod
        log_mod._logo_printed = False

        print_logo()
        out1 = capsys.readouterr().out

        print_logo()
        out2 = capsys.readouterr().out

        # Second call should produce no output
        assert out2 == "" or out2 != out1
