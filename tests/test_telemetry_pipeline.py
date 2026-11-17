"""Exercise real SDK boundaries without contacting model or monitoring services."""

import asyncio
from dataclasses import replace
import io
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from app.core.config import EmbeddingModelConfig, LangfuseConfig, LoggingConfig, OTelConfig
from app.core.errors import ModelProviderError
from app.services.ai.embeddings_runner import OpenAICompatibleEmbeddingClient
from app.services.observability import otel, runtime
from app.services.observability.context import active_otel_ids, get_observability_context, observability_scope
from app.services.observability.langfuse_client import (
    InMemoryLangfuseEventSink, LangfuseObsEvent, LangfuseObsStart, LangfuseSDKEventSink,
    _configure_langfuse, _shutdown_langfuse,
    record_llm_event, flush_langfuse,
    emit_langfuse_event, finish_langfuse_event_scope,
)
from app.services.observability.llm_tracer import llm_tracer
from app.services.observability.log_tracer import log_tracer
from app.services.observability.logging import _configure_logging, log_info
from app.services.observability.queue import InMemoryObservabilityQueue


def _run_isolated(coroutine):
    """Run async tests without displacing Pydantic AI's cached default loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()


class TelemetryPipelineTests(unittest.TestCase):
    def setUp(self):
        self.logs = io.StringIO()
        _configure_logging(LoggingConfig(), stream=self.logs)
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer("test")
        self.enterContext(patch.object(otel, "_otel_tracing_enabled", True))
        self.enterContext(patch.object(otel, "_tracer", self.tracer))
        self.sink = InMemoryLangfuseEventSink()
        _configure_langfuse(LangfuseConfig(), sink=self.sink, synchronous=True)
        self.addCleanup(self.provider.shutdown)
        self.addCleanup(_shutdown_langfuse)
        self.model = SimpleNamespace(provider="test", model_name="test-model", task_type="qa", prompt_version="v1")

    def invoke(self, function):
        return function(model_config=self.model, system_prompt="private system", user_prompt="private user")

    def sdk_sink(self, exporter):
        from langfuse import Langfuse

        config = LangfuseConfig(public_key="test-" + uuid4().hex, secret_key="test-only", host="http://localhost:1")
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))

        def create_client(**kwargs):
            return Langfuse(**kwargs, span_exporter=exporter, httpx_client=client)

        with patch("langfuse.Langfuse", side_effect=create_client):
            sink = LangfuseSDKEventSink(config, service_name="configured-service")
        self.addCleanup(client.close)
        self.addCleanup(sink.shutdown)
        return sink, config

    def test_llm_spans_are_children_and_content_only_reaches_langfuse(self):
        @llm_tracer()
        def call(**kwargs):
            return SimpleNamespace(output="private response", usage=SimpleNamespace(input_tokens=5, output_tokens=3))

        @log_tracer("qa.answer")
        def business():
            self.invoke(call)
            self.invoke(call)

        business()
        spans = self.exporter.get_finished_spans()
        root = spans[-1]
        children = spans[:-1]
        self.assertEqual(len(children), 2)
        for span, event in zip(children, self.sink.events):
            self.assertEqual(span.kind, SpanKind.CLIENT)
            self.assertEqual(span.parent.span_id, root.context.span_id)
            self.assertEqual(event.trace_id, f"{root.context.trace_id:032x}")
            self.assertEqual(event.span_id, f"{span.context.span_id:016x}")
            self.assertEqual(event.response, "private response")
            self.assertTrue(event.usage_available)
            self.assertEqual(event.total_tokens, 8)
            self.assertNotIn("private", str(span.attributes))
        self.assertNotEqual(self.sink.events[0].llm_call_id, self.sink.events[1].llm_call_id)
        self.assertNotIn("private", self.logs.getvalue())
        self.assertNotIn("ainsight.duration_ms", root.attributes)

    def test_missing_usage_and_stream_failure(self):
        @llm_tracer()
        def call(**kwargs):
            return SimpleNamespace(output="answer", usage=SimpleNamespace(input_tokens=0, output_tokens=0))

        self.invoke(call)
        self.assertFalse(self.sink.events[-1].usage_available)
        self.assertIsNone(self.sink.events[-1].input_tokens)

        @llm_tracer()
        def stream(**kwargs):
            yield SimpleNamespace(event="delta", content="partial")
            raise TimeoutError("provider timed out")

        with self.assertRaises(TimeoutError):
            list(self.invoke(stream))
        event = self.sink.events[-1]
        self.assertEqual(event.response, "partial")
        self.assertEqual(event.status, "error")
        self.assertIsNotNone(event.completion_start_time)
        self.assertFalse(event.usage_available)
        self.assertEqual(self.exporter.get_finished_spans()[-1].status.status_code, StatusCode.ERROR)

    def test_point_events_follow_nested_calls_and_expire_with_copied_context(self):
        contexts = []

        @llm_tracer()
        def inner(**kwargs):
            self.assertTrue(record_llm_event("inner_point", result_count=2))
            return SimpleNamespace(output="inner")

        @llm_tracer()
        def outer(**kwargs):
            contexts.append(copy_context())
            self.assertTrue(record_llm_event("outer_before"))
            self.invoke(inner)
            self.assertTrue(record_llm_event("outer_after"))
            return SimpleNamespace(output="outer")

        self.assertFalse(record_llm_event("no_parent"))
        self.invoke(outer)
        self.assertFalse(contexts[0].run(record_llm_event, "late_point"))
        points = self.sink.point_events
        outer_start, inner_start = self.sink.starts
        self.assertEqual([item.llm_call_id for item in points],
                         [outer_start.llm_call_id, inner_start.llm_call_id, outer_start.llm_call_id])
        self.assertEqual(points[1].metadata, {"result_count": 2})
        self.assertEqual(len(self.sink.events), 2)

    def test_late_event_is_skipped_even_while_completion_is_still_queued(self):
        contexts = []

        @llm_tracer()
        def call(**kwargs):
            contexts.append(copy_context())
            return SimpleNamespace(output="answer")

        with patch("app.services.observability.llm_tracer.emit_langfuse_event") as enqueue:
            self.invoke(call)
        enqueue.assert_called_once()
        # The SDK destination still holds the unfinished parent, like a busy
        # background exporter would, but the originating call has already ended.
        self.assertFalse(contexts[0].run(record_llm_event, "late_point"))
        self.assertEqual(self.sink.point_events, [])

    def test_event_after_end_timestamp_is_rejected_before_scope_exit(self):
        exporter = InMemorySpanExporter()
        sink, config = self.sdk_sink(exporter)
        _shutdown_langfuse()
        _configure_langfuse(replace(config, enabled=True), sink=sink)
        contexts = []

        @llm_tracer()
        def call(**kwargs):
            contexts.append(copy_context())
            self.assertTrue(record_llm_event("during_call"))
            return SimpleNamespace(output="answer")

        def enqueue(completion):
            # The model's end time already exists, while its event scope has
            # not exited and the SDK parent is still waiting for completion.
            self.assertFalse(contexts[0].run(record_llm_event, "late_point"))
            return emit_langfuse_event(completion)

        with patch("app.services.observability.llm_tracer.emit_langfuse_event", side_effect=enqueue):
            self.invoke(call)
        self.assertTrue(flush_langfuse())
        point, generation = exporter.get_finished_spans()
        self.assertEqual(point.name, "during_call")
        self.assertEqual(point.parent.span_id, generation.context.span_id)
        self.assertLessEqual(point.start_time, generation.end_time)

    def test_completion_waits_for_an_event_already_being_created(self):
        exporter = InMemorySpanExporter()
        sink, config = self.sdk_sink(exporter)
        _shutdown_langfuse()
        _configure_langfuse(replace(config, enabled=True), sink=sink)
        entered, release, completing = Event(), Event(), Event()
        record_event = sink.record_event
        event_futures = []

        def blocked_event(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release event creation")
            return record_event(*args, **kwargs)

        def finish_events():
            completing.set()
            return finish_langfuse_event_scope()

        @llm_tracer()
        def call(**kwargs):
            event_futures.append(pool.submit(copy_context().run, record_llm_event, "concurrent_point"))
            self.assertTrue(entered.wait(5))
            return SimpleNamespace(output="answer")

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(sink, "record_event", side_effect=blocked_event),
            patch("app.services.observability.llm_tracer.finish_langfuse_event_scope", side_effect=finish_events),
        ):
            call_future = pool.submit(self.invoke, call)
            try:
                self.assertTrue(completing.wait(5))
                self.assertFalse(call_future.done())
            finally:
                release.set()
            self.assertEqual(call_future.result(timeout=5).output, "answer")
            self.assertTrue(event_futures[0].result(timeout=5))
        self.assertTrue(flush_langfuse())
        point, generation = exporter.get_finished_spans()
        self.assertEqual(point.parent.span_id, generation.context.span_id)
        self.assertLessEqual(point.start_time, generation.end_time)

    def test_concurrent_point_events_keep_their_call_identity(self):
        @llm_tracer()
        def call(*, marker, **kwargs):
            self.assertTrue(record_llm_event("checkpoint", marker=marker))
            return SimpleNamespace(output=marker)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda index: call(model_config=self.model, marker=str(index)), range(16)))
        markers = {event.llm_call_id: event.response for event in self.sink.events}
        self.assertEqual(len(results), 16)
        self.assertEqual(len(self.sink.point_events), 16)
        for event in self.sink.point_events:
            self.assertEqual(event.metadata["marker"], markers[event.llm_call_id])

    def test_point_event_failure_and_disabled_capture_do_not_fail_generation(self):
        @llm_tracer()
        def call(**kwargs):
            self.assertFalse(record_llm_event("checkpoint"))
            return SimpleNamespace(output="answer")

        with patch.object(self.sink, "record_event", side_effect=RuntimeError("event sink failed")):
            self.assertEqual(self.invoke(call).output, "answer")
        self.assertEqual(self.sink.events[-1].status, "success")
        _shutdown_langfuse()
        self.assertEqual(self.invoke(call).output, "answer")

    def test_embedding_calls_record_safe_shape_usage_and_client_span(self):
        client = OpenAICompatibleEmbeddingClient(EmbeddingModelConfig(
            enabled=True, model_name="test-embedding", api_key="test-only", max_input_chars=4,
        ))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "data": [{"embedding": [0.12345, 0.67891], "index": 0}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }).encode()
        with self.tracer.start_as_current_span("qa"), patch(
            "app.services.ai.embeddings_runner.urlopen", return_value=response,
        ) as request:
            trace_id, _ = active_otel_ids()
            result = client.embed_texts(["private text"])
        self.assertEqual(result.vectors, [[0.12345, 0.67891]])
        request.assert_called_once()
        event = self.sink.events[-1]
        self.assertEqual(event.observation_type, "embedding")
        self.assertEqual(event.trace_id, trace_id)
        self.assertEqual(event.prompt, {"input_count": 1, "input_chars": 4})
        self.assertEqual(event.response, {"vector_count": 1, "dimensions": 2})
        self.assertEqual(event.input_tokens, 2)
        self.assertEqual(event.total_tokens, 2)
        self.assertIsNone(event.output_tokens)
        self.assertTrue(event.usage_available)
        self.assertNotIn("private", str(event))
        self.assertNotIn("0.12345", str(event))
        span, parent = self.exporter.get_finished_spans()
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.parent.span_id, parent.context.span_id)
        self.assertEqual(span.attributes["gen_ai.operation.name"], "embeddings")

    def test_embedding_usage_unknown_zero_and_failure_are_distinct(self):
        client = OpenAICompatibleEmbeddingClient(EmbeddingModelConfig(
            enabled=True, model_name="test-embedding", api_key="test-only",
        ))
        for usage, available in ((None, False), ({"prompt_tokens": 0, "total_tokens": 0}, True),
                                 ({"total_tokens": 3}, True)):
            with self.subTest(usage=usage):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = json.dumps({
                    "data": [{"embedding": [1.0]}], "usage": usage,
                }).encode()
                with patch("app.services.ai.embeddings_runner.urlopen", return_value=response):
                    client.embed_texts(["text"])
                event = self.sink.events[-1]
                self.assertEqual(event.usage_available, available)
                self.assertIsNone(event.output_tokens)
                self.assertEqual(event.input_tokens, usage.get("prompt_tokens") if usage else None)
        with patch("app.services.ai.embeddings_runner.urlopen", side_effect=TimeoutError("test timeout")):
            with self.assertRaisesRegex(ModelProviderError, "test timeout"):
                client.embed_texts(["text"])
        event = self.sink.events[-1]
        self.assertEqual(event.status, "error")
        self.assertEqual(event.observation_type, "embedding")
        self.assertFalse(event.usage_available)
        self.assertEqual(self.exporter.get_finished_spans()[-1].status.status_code, StatusCode.ERROR)
        count = len(self.sink.events)
        self.assertEqual(client.embed_texts([]).vectors, [])
        self.assertEqual(len(self.sink.events), count)

    def test_embedding_result_survives_langfuse_start_and_export_failures(self):
        client = OpenAICompatibleEmbeddingClient(EmbeddingModelConfig(
            enabled=True, model_name="test-embedding", api_key="test-only",
        ))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "data": [{"embedding": [1.0]}],
        }).encode()
        _shutdown_langfuse()
        with (
            patch.object(self.sink, "begin", side_effect=RuntimeError("SDK unavailable")),
            patch.object(self.sink, "publish", side_effect=RuntimeError("SDK unavailable")),
            patch("app.services.ai.embeddings_runner.urlopen", return_value=response),
        ):
            _configure_langfuse(LangfuseConfig(enabled=True), sink=self.sink)
            try:
                self.assertEqual(client.embed_texts(["text"]).vectors, [[1.0]])
                self.assertTrue(flush_langfuse())
            finally:
                _shutdown_langfuse()

    def test_stream_close_ends_span_and_records_cancelled_once(self):
        @llm_tracer()
        def stream(**kwargs):
            yield SimpleNamespace(event="delta", content="partial")
            yield SimpleNamespace(event="delta", content="unused")

        iterator = self.invoke(stream)
        next(iterator)
        iterator.close()
        self.assertEqual(len(self.sink.events), 1)
        self.assertEqual(self.sink.events[0].status, "cancelled")
        self.assertEqual(len(self.exporter.get_finished_spans()), 1)

    def test_async_stream_close_unwinds_nested_contexts_in_consuming_task(self):
        closed = []

        @log_tracer("nested.stream")
        async def inner():
            try:
                yield "partial"
                yield "unused"
            finally:
                closed.append("inner")

        @log_tracer("qa.stream")
        async def outer():
            from contextlib import aclosing

            async with aclosing(inner()) as stream:
                async for item in stream:
                    yield item

        async def consume():
            original = get_observability_context()
            stream = outer()
            self.assertEqual(await anext(stream), "partial")
            await stream.aclose()
            self.assertEqual(get_observability_context(), original)
            self.assertEqual(active_otel_ids(), (None, None))
            self.assertEqual(closed, ["inner"])

        _run_isolated(consume())
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(span.attributes["ainsight.status"] == "cancelled" for span in spans))

    def test_http_parent_is_error_even_when_business_exception_already_logged(self):
        for response_started in (False, True):
            with self.subTest(response_started=response_started):
                self.exporter.clear()
                self.logs.seek(0)
                self.logs.truncate()

                @log_tracer("qa.answer")
                async def app(scope, receive, send):
                    if response_started:
                        await send({"type": "http.response.start", "status": 200})
                    raise RuntimeError("business failure")

                async def receive():
                    return {"type": "http.request"}

                async def send(message):
                    pass

                with self.assertRaisesRegex(RuntimeError, "business failure"):
                    _run_isolated(otel.ObservabilityMiddleware(app)(
                        {"type": "http", "method": "POST", "path": "/qa", "headers": []}, receive, send,
                    ))
                spans = self.exporter.get_finished_spans()
                self.assertEqual(len(spans), 2)
                self.assertTrue(all(span.status.status_code == StatusCode.ERROR for span in spans))
                parent = spans[-1]
                self.assertEqual(parent.kind, SpanKind.SERVER)
                self.assertEqual(parent.attributes["error.type"], "RuntimeError")
                self.assertEqual(parent.attributes.get("http.response.status_code"), 200 if response_started else None)
                logs = [json.loads(line) for line in self.logs.getvalue().splitlines()]
                self.assertEqual(sum(item["event"] == "business_error" for item in logs), 1)

    def test_log_length_limit_preserves_trace_and_business_identifiers(self):
        _configure_logging(LoggingConfig(max_field_length=20), stream=self.logs)
        request_id = "i_" + "r" * 32
        with self.tracer.start_as_current_span("business"), observability_scope(request_id=request_id):
            trace_id, span_id = active_otel_ids()
            log_info("checked", description="x" * 100)
        payload = json.loads(self.logs.getvalue())
        self.assertEqual(payload["trace_id"], trace_id)
        self.assertEqual(payload["span_id"], span_id)
        self.assertEqual(payload["request_id"], request_id)
        self.assertIn("[truncated]", payload["description"])

    def test_langfuse_sdk_uses_isolated_roots_and_keeps_real_end_timestamp(self):
        lf_exporter = InMemorySpanExporter()
        sink, config = self.sdk_sink(lf_exporter)
        with self.tracer.start_as_current_span("business"):
            trace_id, span_id = active_otel_ids()
            start = LangfuseObsStart(trace_id, "llm_test", "qa", "QAAgent", "test", "test-model", "v1",
                                     {"system": "original prompt", "user": "question"}, time.time_ns(), span_id=span_id)
            sink.begin(start)
            before_event = time.time_ns()
            self.assertTrue(sink.record_event("llm_test", "checkpoint", {"phase": "received"}))
            after_event = time.time_ns()
            ended = time.time_ns()
            event = LangfuseObsEvent(
                trace_id, "llm_test", "qa", "QAAgent", "test", "test-model", "v1", start.prompt,
                "original response", None, None, None, 1, "success", start.started_at_ns, ended,
                usage_available=False, span_id=span_id,
            )
            sink.publish(event)
        sink.flush()
        generations = lf_exporter.get_finished_spans()
        self.assertEqual(len(generations), 2)
        point, generation = generations
        self.assertEqual(point.name, "checkpoint")
        self.assertEqual(point.attributes["langfuse.observation.type"], "event")
        self.assertEqual(point.parent.span_id, generation.context.span_id)
        self.assertEqual(point.context.trace_id, generation.context.trace_id)
        self.assertEqual(point.start_time, point.end_time)
        self.assertLessEqual(before_event, point.start_time)
        self.assertLessEqual(point.start_time, after_event)
        self.assertFalse(sink.record_event("llm_test", "too_late", {}))
        self.assertEqual(generation.context.trace_id, int(trace_id, 16))
        self.assertIsNone(generation.parent)
        self.assertEqual(generation.end_time, ended)
        self.assertEqual(generation.resource.attributes["service.name"], "configured-service")
        self.assertIn("original prompt", str(generation.attributes))
        self.assertIn("original response", str(generation.attributes))
        self.assertFalse(any("usage_details" in key for key in generation.attributes))
        self.assertEqual([span.name for span in self.exporter.get_finished_spans()], ["business"])
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            LangfuseSDKEventSink(config, service_name="configured-service")
        sink.shutdown()
        # The real SDK retains a stopped resource manager under the same key.
        # Reject reconstruction explicitly rather than silently accepting spans.
        with self.assertRaisesRegex(RuntimeError, "restart the process"):
            LangfuseSDKEventSink(config, service_name="configured-service")

    def test_embedding_and_point_events_reach_real_sdk_through_async_delivery(self):
        exporter = InMemorySpanExporter()
        sink, config = self.sdk_sink(exporter)
        _shutdown_langfuse()
        _configure_langfuse(replace(config, enabled=True), sink=sink)
        client = OpenAICompatibleEmbeddingClient(EmbeddingModelConfig(
            enabled=True, model_name="test-embedding", api_key="test-only",
        ))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "data": [{"embedding": [0.12345, 0.67891]}],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }).encode()

        def respond(*args, **kwargs):
            self.assertTrue(record_llm_event("embedding_response_received", vector_count=1))
            return response

        with self.tracer.start_as_current_span("qa"), patch(
            "app.services.ai.embeddings_runner.urlopen", side_effect=respond,
        ):
            trace_id, _ = active_otel_ids()
            client.embed_texts(["private text"])
        self.assertTrue(flush_langfuse())
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        point, embedding = spans
        self.assertEqual(embedding.attributes["langfuse.observation.type"], "embedding")
        self.assertEqual(embedding.context.trace_id, int(trace_id, 16))
        self.assertEqual(point.parent.span_id, embedding.context.span_id)
        self.assertIn('"input": 0', str(embedding.attributes))
        self.assertIn('"total": 0', str(embedding.attributes))
        self.assertNotIn('"output": 0', str(embedding.attributes))
        self.assertNotIn("private text", str(spans))
        self.assertNotIn("0.12345", str(spans))
        self.assertEqual(len(self.exporter.get_finished_spans()), 2)

    def test_http_metrics_use_seconds_and_only_low_cardinality_attributes(self):
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        self.addCleanup(provider.shutdown)
        histogram = provider.get_meter("test").create_histogram("http.server.request.duration", unit="s")
        with patch.object(otel, "_request_duration", histogram):
            otel._record_request_metrics("GET", "/users/{id}", 200, 0.125, "https")
            otel._record_request_metrics("GET", "/users/{id}", 503, 0.25, "https")
        metric = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics[0]
        self.assertEqual(metric.name, "http.server.request.duration")
        self.assertEqual(metric.unit, "s")
        points = metric.data.data_points
        self.assertEqual(sum(point.count for point in points), 2)
        self.assertAlmostEqual(sum(point.sum for point in points), 0.375)
        self.assertEqual(points[1].attributes["error.type"], "503")
        self.assertNotIn("trace_id", points[0].attributes)

    def test_queue_losses_are_counted_once_without_an_active_span(self):
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        self.addCleanup(provider.shutdown)
        counter = provider.get_meter("test").create_counter("ainsight.langfuse.queue.dropped", unit="{event}")
        started, release = Event(), Event()

        def consume(event):
            started.set()
            release.wait(1)
            if event == "fail":
                raise RuntimeError("sink failed")

        with patch.object(otel, "_langfuse_dropped", counter):
            queue = InMemoryObservabilityQueue(consume, max_size=1, enqueue_timeout_ms=1,
                drop_when_full=True, name="test-losses", on_drop=lambda event, reason: otel.record_langfuse_drop(reason))
            try:
                self.assertTrue(queue.enqueue("fail"))
                self.assertTrue(started.wait(1))
                self.assertTrue(queue.enqueue("ok"))
                self.assertFalse(queue.enqueue("overflow"))
            finally:
                release.set()
                self.assertTrue(queue.shutdown(2))
            self.assertFalse(queue.enqueue("stopped"))
        metric = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics[0]
        self.assertEqual({point.attributes["reason"]: point.value for point in metric.data.data_points},
                         {"queue_full": 1, "export_failed": 1, "queue_stopped": 1})

    def test_shutdown_is_idempotent_and_flushes_otel_after_langfuse_failure(self):
        with (
            patch.object(runtime, "running", True),
            patch.object(runtime, "_shutdown_langfuse", side_effect=RuntimeError("flush failed")) as lf,
            patch.object(runtime, "_shutdown_otel") as ot,
        ):
            runtime.shutdown_observability()
            runtime.shutdown_observability()
            lf.assert_called_once()
            ot.assert_called_once()

    def test_partial_otel_startup_releases_exporter_after_processor_failure(self):
        config = OTelConfig(enabled=True)
        config = replace(config, queue=replace(config.queue, max_queue_size=0))
        with patch.object(otel, "_build_trace_exporter", return_value=self.exporter):
            with patch.object(self.exporter, "shutdown", wraps=self.exporter.shutdown) as close:
                try:
                    with self.assertRaises(ValueError):
                        otel._configure_otel(config, service_name="offline-test")
                finally:
                    otel._shutdown_otel()
                close.assert_called_once()
                self.assertIsNone(otel._tracer_provider)

    def test_langfuse_constructor_keeps_original_error_when_cleanup_fails(self):
        original = ValueError("SDK initialization failed")
        provider = MagicMock()
        provider.shutdown.side_effect = RuntimeError("cleanup failed")
        with (
            patch("app.services.observability.langfuse_client.TracerProvider", return_value=provider) as create,
            patch("langfuse.Langfuse", side_effect=original),
        ):
            with self.assertRaises(ValueError) as caught:
                LangfuseSDKEventSink(
                    LangfuseConfig(public_key="test-" + uuid4().hex, secret_key="test-only"),
                    service_name="offline-test",
                )
            self.assertIs(caught.exception, original)
            provider.shutdown.assert_called_once()
            self.assertNotIn("shutdown_on_exit", create.call_args.kwargs)

    def test_otel_providers_keep_sdk_exit_hooks_and_cancel_them_on_shutdown(self):
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        config = OTelConfig(enabled=True)
        config = replace(config, metrics=replace(config.metrics, enabled=True))
        with (
            patch.object(otel, "_build_trace_exporter", return_value=self.exporter),
            patch.object(otel, "_build_metric_exporter", return_value=MagicMock()),
            patch("opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader", return_value=InMemoryMetricReader()),
        ):
            try:
                otel._configure_otel(config, service_name="offline-test")
                tracer, meter = otel._tracer_provider, otel._meter_provider
                self.assertIsNotNone(tracer._atexit_handler)
                self.assertIsNotNone(meter._atexit_handler)
            finally:
                otel._shutdown_otel()
            self.assertIsNone(tracer._atexit_handler)
            self.assertIsNone(meter._atexit_handler)


if __name__ == "__main__":
    unittest.main()
