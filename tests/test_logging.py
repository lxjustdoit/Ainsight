import io
import json
import unittest

from app.core.config import LoggingConfig
from app.services.observability.context import observability_scope
from app.services.observability.logging import _configure_logging, log_error, log_info


class StructuredLoggingTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        _configure_logging(LoggingConfig(level="INFO", max_field_length=2000), stream=self.stream)

    def records(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines() if line]

    def test_info_log_injects_context_and_keeps_debug_content_while_redaction_is_paused(self):
        with observability_scope(trace_id="00000000000070008000000000000001", user_id="user-1"):
            log_info(
                "request_finished",
                duration_ms=12,
                apiKey="secret-value",
                access_token="access-value",
                **{"x-api-key": "header-secret", "trace_id": "client-controlled"},
                question="must not enter service logs",
            )

        record = self.records()[0]
        self.assertEqual(record["event"], "request_finished")
        self.assertEqual(record["trace_id"], "00000000000070008000000000000001")
        self.assertEqual(record["user_id"], "user-1")
        self.assertEqual(record["apiKey"], "secret-value")
        self.assertEqual(record["access_token"], "access-value")
        self.assertEqual(record["x-api-key"], "header-secret")
        self.assertEqual(record["question"], "must not enter service logs")

    def test_error_log_has_status_and_stacktrace(self):
        try:
            raise RuntimeError("provider failed Authorization: Bearer private-token")
        except RuntimeError as exc:
            log_error("dependency_error", exc, dependency="llm_provider")

        record = self.records()[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertIn("RuntimeError: provider failed", record["stacktrace"])
        self.assertIn("private-token", record["error_message"])
        self.assertIn("private-token", record["stacktrace"])

if __name__ == "__main__":
    unittest.main()
