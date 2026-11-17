import asyncio
import io
import json
from threading import Event, Lock
from time import sleep
import unittest
from unittest.mock import MagicMock, patch

from app.core.config import LangfuseConfig, LoggingConfig
from app.services.observability.context import increment_llm_call_count, observability_scope, update_observability_context
from app.services.observability import langfuse_client
from app.services.observability.langfuse_client import (
    LangfuseObsEvent,
    LangfuseObsStart,
    LangfuseSDKEventSink,
    _configure_langfuse,
)
from app.services.observability.log_tracer import log_tracer
from app.services.observability.logging import _configure_logging
from app.services.observability.otel import ObservabilityMiddleware
from app.services.observability import otel
from app.services.observability.queue import InMemoryObservabilityQueue


class _StreamItem:
    def __init__(self, event):
        self.event = event


class _FakeObservation:
    def __init__(self):
        self.update_fields = {}
        self.end_time = None

    def update(self, **fields):
        self.update_fields = fields

    def end(self, *, end_time=None):
        self.end_time = end_time


class _FakeLangfuseClient:
    def __init__(self):
        self.observation = _FakeObservation()
        self.start_fields = {}

    def start_observation(self, **fields):
        self.start_fields = fields
        return self.observation


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        _configure_logging(LoggingConfig(), stream=self.stream)

    def records(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines() if line]

    def test_log_tracer_records_business_entry_and_completion(self):
        @log_tracer("test.operation", task_type="test", path="/test")
        def operation():
            return "done"

        with observability_scope(trace_id="00000000000070008000000000000011"):
            self.assertEqual(operation(), "done")

        records = self.records()
        self.assertEqual([record["event"] for record in records], ["operation_started", "operation_finished"])
        self.assertTrue(all(record["trace_id"] == "00000000000070008000000000000011" for record in records))

    def test_log_tracer_marks_handled_stream_error_as_failed(self):
        @log_tracer("test.stream", task_type="test", path="/test/stream")
        def operation():
            yield _StreamItem("error")
            yield _StreamItem("done")

        list(operation())

        finished = [record for record in self.records() if record["event"] == "operation_finished"][-1]
        self.assertEqual(finished["status"], "failed")

    def test_log_tracer_supports_async_business_operations(self):
        @log_tracer("test.async", task_type="test", path="/test/async")
        async def operation():
            return "done"

        self.assertEqual(asyncio.run(operation()), "done")
        records = self.records()
        self.assertEqual([record["event"] for record in records], ["operation_started", "operation_finished"])

    def test_log_tracer_supports_async_streams(self):
        @log_tracer("test.async_stream", task_type="test", path="/test/async-stream")
        async def operation():
            yield _StreamItem("done")

        async def consume():
            return [item async for item in operation()]

        self.assertEqual(len(asyncio.run(consume())), 1)
        self.assertEqual(self.records()[-1]["status"], "success")

    def test_observability_queue_delivers_events_off_call_path(self):
        delivered = []
        queue = InMemoryObservabilityQueue(
            delivered.append,
            max_size=4,
            enqueue_timeout_ms=1,
            drop_when_full=True,
            name="test-observability-queue",
        )

        self.assertTrue(queue.enqueue("event"))
        self.assertTrue(queue.flush(1.0))
        queue.shutdown(1.0)

        self.assertEqual(delivered, ["event"])

    def test_queue_does_not_report_shutdown_while_consumer_is_running(self):
        started = Event()
        release = Event()

        def consume(_event):
            started.set()
            release.wait(1.0)

        queue = InMemoryObservabilityQueue(
            consume,
            max_size=1,
            enqueue_timeout_ms=1,
            drop_when_full=True,
            name="test-slow-observability-queue",
        )
        queue.enqueue("event")
        self.assertTrue(started.wait(1.0))
        self.assertFalse(queue.shutdown(0.01))
        release.set()
        sleep(0.02)

    def test_asgi_middleware_adds_trace_header_and_correlated_request_log(self):
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/qa",
            "headers": [
                (b"idempotency-key", b"i_test-request"),
                (b"x-request-id", b"gateway-request"),
                (b"traceparent", b"00-11111111111111111111111111111111-2222222222222222-01"),
            ],
        }
        asyncio.run(ObservabilityMiddleware(app)(scope, receive, send))

        start = sent[0]
        trace_headers = [value for key, value in start["headers"] if key == b"x-trace-id"]
        self.assertEqual(len(trace_headers), 1)
        self.assertRegex(trace_headers[0].decode("ascii"), r"^[0-9a-f]{32}$")
        self.assertNotEqual(trace_headers[0].decode("ascii"), "11111111111111111111111111111111")
        request_log = [record for record in self.records() if record["event"] == "request_finished"][-1]
        self.assertEqual(request_log["request_id"], "i_test-request")
        self.assertEqual(request_log["external_request_id"], "gateway-request")

    def test_request_log_receives_operation_context_updates(self):
        sent = []

        @log_tracer("test.request_operation", task_type="qa", path="/api/qa")
        async def business_operation():
            update_observability_context(ctx="context-hash", conversation_id="conversation-1")
            increment_llm_call_count()

        async def app(scope, receive, send):
            await business_operation()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/api/qa", "headers": []}
        asyncio.run(ObservabilityMiddleware(app)(scope, receive, send))

        request_log = [record for record in self.records() if record["event"] == "request_finished"][-1]
        self.assertEqual(request_log["ctx"], "context-hash")
        self.assertEqual(request_log["conversation_id"], "conversation-1")
        self.assertEqual(request_log["llm_call_count"], 1)

    def test_http_error_marks_server_span(self):
        span = MagicMock()
        span.is_recording.return_value = True
        with patch.object(otel, "_otel_tracing_enabled", True), patch.object(otel, "_current_span", return_value=span):
            otel._record_http_span_status(503, method="GET", route_path="/users/{user_id}")

        span.set_attribute.assert_any_call("http.response.status_code", 503)
        span.set_attribute.assert_any_call("http.route", "/users/{user_id}")
        span.update_name.assert_called_once_with("GET /users/{user_id}")
        span.set_status.assert_called_once()

    def test_span_error_keeps_debug_fields_while_redaction_is_paused(self):
        span = MagicMock()
        span.is_recording.return_value = True
        error = RuntimeError("provider failed Authorization: Bearer private-token")

        with patch.object(otel, "_otel_tracing_enabled", True), patch.object(otel, "_current_span", return_value=span):
            otel.record_span_error(
                "provider.error",
                error,
                dependency="llm_provider",
                authorization="Bearer attribute-token",
            )

        serialized_calls = str(span.mock_calls)
        self.assertIn("private-token", serialized_calls)
        self.assertIn("attribute-token", serialized_calls)
        self.assertNotIn("[REDACTED]", serialized_calls)
        span.record_exception.assert_not_called()

    def test_log_tracer_exports_one_debug_exception_event_while_redaction_is_paused(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("ainsight-test")

        @log_tracer("test.secret_failure", task_type="test", path="/test")
        def failing_operation():
            raise RuntimeError("provider failed Authorization: Bearer private-token")

        with patch.object(otel, "_otel_tracing_enabled", True), patch.object(otel, "_tracer", tracer):
            with self.assertRaises(RuntimeError):
                failing_operation()

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        serialized_span = str(spans[0].events) + str(spans[0].status) + str(spans[0].attributes)
        self.assertIn("private-token", serialized_span)
        self.assertNotIn("[REDACTED]", serialized_span)
        self.assertEqual([event.name for event in spans[0].events], ["exception"])
        provider.shutdown()

    def test_internal_langfuse_startup_does_not_close_resources(self):
        old_sink = MagicMock()
        old_queue = MagicMock()
        old_queue.shutdown.return_value = False
        replacement_sink = MagicMock()

        with (
            patch.object(langfuse_client, "_event_sink", old_sink),
            patch.object(langfuse_client, "_event_queue", old_queue),
            patch.object(langfuse_client, "_capture_enabled", True),
        ):
            self.assertIs(_configure_langfuse(LangfuseConfig(), sink=replacement_sink, synchronous=True), replacement_sink)
            self.assertIs(langfuse_client._event_sink, replacement_sink)
            old_queue.shutdown.assert_not_called()
            old_sink.flush.assert_not_called()
            old_sink.shutdown.assert_not_called()

    def test_langfuse_adapter_preserves_actual_end_time_and_standard_usage(self):
        sink = LangfuseSDKEventSink.__new__(LangfuseSDKEventSink)
        sink._client = _FakeLangfuseClient()
        sink._observations = {}
        sink._observations_lock = Lock()
        start = LangfuseObsStart(
            trace_id="00000000000070008000000000000001",
            llm_call_id="llm_00000000000070008000000000000001",
            task_type="qa",
            agent_name="QAAgent",
            provider="test",
            model_name="test-model",
            prompt_version="qa_v1",
            prompt={"system": "system", "user": "question"},
            started_at_ns=100,
        )
        sink.begin(start)
        sink.publish(
            LangfuseObsEvent(
                trace_id=start.trace_id,
                llm_call_id=start.llm_call_id,
                task_type=start.task_type,
                agent_name=start.agent_name,
                provider=start.provider,
                model_name=start.model_name,
                prompt_version=start.prompt_version,
                prompt=start.prompt,
                response="answer",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                latency_ms=20,
                status="success",
                started_at_ns=100,
                completed_at_ns=20_000_100,
            )
        )

        observation = sink._client.observation
        self.assertEqual(observation.end_time, 20_000_100)
        self.assertEqual(observation.update_fields["usage_details"], {"input": 10, "output": 5, "total": 15})
        self.assertEqual(observation.update_fields["metadata"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
