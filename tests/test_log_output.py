"""File/stdout logging and configuration contracts without local credentials."""

import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import LoggingConfig, ObservabilityConfig, load_config
from app.services.observability.context import observability_scope
from app.services.observability.logging import _configure_logging, get_logger, log_error, log_info


class LogOutputTests(unittest.TestCase):
    def setUp(self):
        self.logger = get_logger()
        self.previous_handlers = list(self.logger.handlers)
        self.previous_level = self.logger.level
        self.previous_propagate = self.logger.propagate
        for handler in self.previous_handlers:
            self.logger.removeHandler(handler)

    def tearDown(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        for handler in self.previous_handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(self.previous_level)
        self.logger.propagate = self.previous_propagate

    def test_unconfigured_and_empty_paths_use_stdout(self):
        for output_path in (None, ""):
            with self.subTest(output_path=output_path), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                _configure_logging(LoggingConfig(output_path=output_path))
                log_info("stdout_event")
                record = json.loads(stdout.getvalue())
                self.assertEqual(record["event"], "stdout_event")
                self.assertEqual(record["service_name"], ObservabilityConfig.service_name)

    def test_file_output_creates_parents_appends_and_takes_precedence_over_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory, "logs", "service.jsonl")
            config = LoggingConfig(output_path=str(output_path))
            stream = io.StringIO()
            _configure_logging(config, service_name="file-service", stream=stream)
            log_info("first", text="\u6d4b\u8bd5")
            old_handler = self.logger.handlers[0]
            old_stream = old_handler.stream
            _configure_logging(config, service_name="file-service", stream=stream)
            self.assertTrue(old_stream.closed)
            self.assertIsNone(old_handler.stream)
            log_info("second")
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["event"] for record in records], ["first", "second"])
            self.assertEqual(records[0]["text"], "\u6d4b\u8bd5")
            self.assertTrue(all(record["service_name"] == "file-service" for record in records))
            self.assertEqual(stream.getvalue(), "")
            self.assertEqual(len(self.logger.handlers), 1)

    def test_reconfiguration_switches_files_then_stdout_without_closing_external_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory, "first.jsonl")
            second_path = Path(directory, "second.jsonl")
            external_path = Path(directory, "external.log")
            external = logging.FileHandler(external_path, encoding="utf-8")
            self.logger.addHandler(external)
            with patch.object(external, "close", wraps=external.close) as close:
                _configure_logging(LoggingConfig(output_path=str(first_path)))
                log_info("first")
                _configure_logging(LoggingConfig(output_path=str(second_path)))
                log_info("second")
                stream = io.StringIO()
                _configure_logging(LoggingConfig(), stream=stream)
                log_info("third")
                close.assert_not_called()
                self.assertIn(external, self.logger.handlers)
                self.assertFalse(external.stream.closed)
                self.assertEqual(external_path.read_text().splitlines(), ["first", "second", "third"])
            self.assertEqual(json.loads(first_path.read_text())["event"], "first")
            self.assertEqual(json.loads(second_path.read_text())["event"], "second")
            self.assertEqual(json.loads(stream.getvalue())["event"], "third")

    def test_reconfiguration_does_not_close_caller_stream_or_duplicate_records(self):
        stream = io.StringIO()
        for index in range(3):
            _configure_logging(LoggingConfig(), stream=stream)
            log_info(f"event_{index}")
        self.assertFalse(stream.closed)
        self.assertEqual(len(stream.getvalue().splitlines()), 3)
        self.assertEqual(len(self.logger.handlers), 1)

    def test_failed_file_setup_leaves_previous_handler_and_level_usable(self):
        stream = io.StringIO()
        _configure_logging(LoggingConfig(), stream=stream)
        handler = self.logger.handlers[0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                _configure_logging(LoggingConfig(level="ERROR", output_path=directory))
        self.assertEqual(self.logger.handlers, [handler])
        self.assertEqual(self.logger.level, logging.INFO)
        log_info("still_usable")
        self.assertEqual(json.loads(stream.getvalue())["event"], "still_usable")

    def test_shared_service_name_cannot_be_overridden_and_is_not_truncated(self):
        stream = io.StringIO()
        _configure_logging(LoggingConfig(max_field_length=4), service_name="shared-service", stream=stream)
        log_info("test", service_name="caller-value")
        get_logger("child").warning("test", extra={"service_name": "record-value"})
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([record["service_name"] for record in records], ["shared-service"] * 2)
        _configure_logging(LoggingConfig(), service_name="new-service", stream=stream)
        log_info("new")
        self.assertEqual(json.loads(stream.getvalue().splitlines()[-1])["service_name"], "new-service")

    def test_file_logs_keep_trace_context_and_full_exception_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory, "errors.jsonl")
            _configure_logging(LoggingConfig(max_field_length=20, output_path=str(output_path)))
            with observability_scope(trace_id="test-trace", user_id="test-user"):
                try:
                    try:
                        raise ValueError("original failure " + "x" * 2100)
                    except ValueError as cause:
                        raise RuntimeError("final failure") from cause
                except RuntimeError as error:
                    log_error("failed", error, detail="d" * 100)
                    get_logger().exception("standard_exception")
            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertEqual(record["trace_id"], "test-trace")
                self.assertEqual(record["user_id"], "test-user")
                self.assertIn("ValueError: original failure " + "x" * 2100, record["stacktrace"])
                self.assertIn("RuntimeError: final failure", record["stacktrace"])
                self.assertIn("Traceback (most recent call last)", record["stacktrace"])
                self.assertNotIn("[truncated]", record["stacktrace"])
            self.assertEqual(records[0]["detail"], "d" * 20 + "...[truncated]")


class LogOutputConfigTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_output_path_defaults_and_null_or_empty_values(self):
        self.assertIsNone(LoggingConfig().output_path)
        for settings in ({}, {"output_path": None}, {"output_path": ""}):
            with self.subTest(settings=settings), tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
                json.dump({"observability": {"logging": settings}}, config_file)
                config_file.flush()
                self.assertIsNone(load_config(config_file.name).observability.logging.output_path)

    def test_yaml_and_json_paths_and_environment_precedence(self):
        for suffix in (".json", ".yaml"):
            with self.subTest(suffix=suffix), tempfile.NamedTemporaryFile("w", suffix=suffix) as config_file:
                # JSON is also valid YAML; exercise both loader branches with one fixture.
                json.dump({"observability": {"service_name": "file-service", "logging": {
                    "level": "DEBUG", "max_field_length": 1500, "output_path": "logs/file.jsonl",
                }}}, config_file)
                config_file.flush()
                config = load_config(config_file.name)
                self.assertEqual(config.observability.logging.output_path, "logs/file.jsonl")
                self.assertEqual(config.observability.service_name, "file-service")
                with patch.dict(os.environ, {"LOG_OUTPUT_PATH": "logs/env.jsonl", "SERVICE_NAME": "env-service"}):
                    config = load_config(config_file.name)
                self.assertEqual(config.observability.logging.output_path, "logs/env.jsonl")
                self.assertEqual(config.observability.service_name, "env-service")
                self.assertEqual(config.observability.logging.level, "DEBUG")
                self.assertEqual(config.observability.logging.max_field_length, 1500)
                with patch.dict(os.environ, {"LOG_OUTPUT_PATH": ""}):
                    self.assertIsNone(load_config(config_file.name).observability.logging.output_path)

    def test_overlay_preserves_siblings_and_explicit_file_bypasses_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory, "base.json")
            override = Path(directory, "override.json")
            base.write_text(json.dumps({"observability": {
                "service_name": "base-service",
                "logging": {"level": "DEBUG", "max_field_length": 1000, "output_path": "base.jsonl"},
            }}), encoding="utf-8")
            override.write_text(json.dumps({"observability": {"logging": {"output_path": "override.jsonl"}}}), encoding="utf-8")
            # Discover only synthetic fixtures; never inspect the real local config.
            with patch("app.core.config._first_existing", side_effect=[str(base), str(override)]):
                config = load_config()
            self.assertEqual(config.observability.service_name, "base-service")
            self.assertEqual(config.observability.logging.level, "DEBUG")
            self.assertEqual(config.observability.logging.max_field_length, 1000)
            self.assertEqual(config.observability.logging.output_path, "override.jsonl")
            with patch("app.core.config._first_existing") as discover:
                config = load_config(str(base))
                discover.assert_not_called()
            self.assertEqual(config.observability.logging.output_path, "base.jsonl")
            with patch.dict(os.environ, {"AINSIGHT_CONFIG_FILE": str(override)}):
                with patch("app.core.config._first_existing") as discover:
                    config = load_config()
                    discover.assert_not_called()
            self.assertEqual(config.observability.logging.output_path, "override.jsonl")


if __name__ == "__main__":
    unittest.main()
