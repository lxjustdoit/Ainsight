"""Offline SDK queue metrics checks through the application's OTel lifecycle."""

from dataclasses import replace
import os
from threading import Event, Lock
import unittest
from unittest.mock import patch

from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from app.core.config import ObservabilityConfig, OTelConfig
from app.services.observability import otel, runtime


PREFIX = "otel.sdk.processor.span."
FLAG = "OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED"


class _MetricSink(MetricExporter):
    """Capture real periodic-reader exports, including its final shutdown batch."""

    def __init__(self):
        super().__init__()
        self.batches = []
        self.closed = False

    def export(self, metrics_data, timeout_millis=10000, **kwargs):
        self.batches.append(metrics_data)
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis=10000):
        return True

    def shutdown(self, timeout_millis=30000, **kwargs):
        self.closed = True


class _SpanSink(SpanExporter):
    """Hold a batch in flight while the producer fills the bounded queue."""

    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.names = []
        self.closed = False
        self.error = None

    def export(self, spans):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("Test did not release span exporter")
        if self.error:
            raise self.error
        self.names.extend(span.name for span in spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.closed = True


class ProcessorMetricsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            # Keep the real reader worker (and its shutdown collection) but
            # avoid periodic collection racing deterministic assertions.
            FLAG: "true", "OTEL_METRIC_EXPORT_INTERVAL": "3600000",
        }))
        self.enterContext(patch.multiple(
            otel, _tracer=None, _meter=None, _tracer_provider=None,
            _meter_provider=None, _request_duration=None, _langfuse_dropped=None,
            _otel_tracing_enabled=False, _otel_metrics_enabled=False,
        ))
        self.spans = _SpanSink()
        self.metrics = _MetricSink()
        self.enterContext(patch.object(otel, "_build_trace_exporter", return_value=self.spans))
        self.enterContext(patch.object(otel, "_build_metric_exporter", return_value=self.metrics))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.addCleanup(otel._shutdown_otel)
        self.addCleanup(self.spans.release.set)
        config = OTelConfig(enabled=True)
        self.config = replace(
            config, metrics=replace(config.metrics, enabled=True),
            queue=replace(config.queue, max_queue_size=2, max_export_batch_size=1,
                          schedule_delay_ms=60000),
        )

    def start(self, config=None):
        otel._configure_otel(config or self.config, service_name="processor-metrics-test")

    def emit(self, name="test"):
        with otel.start_span(name):
            pass

    def collect(self):
        self.assertTrue(otel._meter_provider.force_flush())
        return self.points(self.metrics.batches[-1])

    def points(self, batch):
        points = {}
        for resource in batch.resource_metrics:
            self.assertEqual(resource.resource.attributes["service.name"], "processor-metrics-test")
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    points.setdefault(metric.name, []).extend(metric.data.data_points)
        return points

    def test_queue_size_capacity_drop_and_drain_use_real_sdk(self):
        self.start()
        self.spans.release.clear()
        self.emit("in-flight")
        self.assertTrue(self.spans.entered.wait(5))
        self.assertEqual(self.collect()[PREFIX + "queue.size"][0].value, 0)
        for name in ("oldest", "second", "newest"):
            self.emit(name)
        points = self.collect()
        self.assertEqual(points[PREFIX + "queue.size"][0].value, 2)
        self.assertEqual(points[PREFIX + "queue.capacity"][0].value, 2)
        dropped = points[PREFIX + "processed"]
        self.assertEqual([(p.attributes.get("error.type"), p.value) for p in dropped], [("queue_full", 1)])
        self.assertIn("otel.component.name", dropped[0].attributes)
        self.spans.release.set()
        self.assertTrue(otel._tracer_provider.force_flush())
        points = self.collect()
        self.assertEqual(points[PREFIX + "queue.size"][0].value, 0)
        self.assertEqual({p.attributes.get("error.type"): p.value for p in points[PREFIX + "processed"]},
                         {"queue_full": 1, None: 3})
        self.assertEqual(self.spans.names, ["in-flight", "second", "newest"])
        self.assertIn("otel.sdk.metric_reader.collection.duration", points)

    def test_missing_flag_preserves_application_metrics_and_traces(self):
        os.environ.pop(FLAG, None)
        self.start()
        self.emit()
        otel.record_langfuse_drop("queue_full")
        self.assertTrue(otel._tracer_provider.force_flush())
        self.collect()
        points = self.collect()
        self.assertFalse(any(name.startswith("otel.sdk.") for name in points))
        self.assertEqual(points["ainsight.langfuse.queue.dropped"][0].value, 1)
        self.assertEqual(self.spans.names, ["test"])

    def test_metrics_disabled_does_not_bind_to_global_provider(self):
        config = replace(self.config, metrics=replace(self.config.metrics, enabled=False))
        with patch("opentelemetry.sdk.trace.export.get_meter_provider", side_effect=AssertionError("Global provider")):
            self.start(config)
        self.emit()
        self.assertTrue(otel._tracer_provider.force_flush())
        self.assertIsNone(otel._meter_provider)
        self.assertEqual(self.spans.names, ["test"])

    def test_traces_disabled_keeps_business_metrics_without_processor_metrics(self):
        self.start(replace(self.config, traces=replace(self.config.traces, enabled=False)))
        otel.record_langfuse_drop("queue_full")
        points = self.collect()
        self.assertIsNone(otel._tracer_provider)
        self.assertFalse(any(name.startswith(PREFIX) for name in points))
        self.assertEqual(points["ainsight.langfuse.queue.dropped"][0].value, 1)

    def test_shutdown_collects_final_processed_count_after_draining_spans(self):
        self.start(replace(self.config, queue=replace(self.config.queue, max_export_batch_size=2)))
        self.emit("pending-at-shutdown")
        otel._shutdown_otel()
        points = self.points(self.metrics.batches[-1])
        self.assertEqual(points[PREFIX + "processed"][0].value, 1)
        self.assertEqual(points[PREFIX + "queue.size"][0].value, 0)
        self.assertTrue(self.metrics.closed)
        self.assertTrue(self.spans.closed)

    def test_export_exception_is_distinct_from_queue_full(self):
        self.start()
        self.spans.error = ConnectionError("offline test failure")
        self.emit()
        self.assertTrue(otel._tracer_provider.force_flush())
        points = self.collect()[PREFIX + "processed"]
        self.assertEqual([(p.attributes.get("error.type"), p.value) for p in points], [("ConnectionError", 1)])

    def test_runtime_cleans_metrics_when_trace_processor_creation_fails(self):
        failure = ValueError("processor construction failed")
        with (
            patch.multiple(runtime, running=None, _active_config=None, _active_sink=None, _lifecycle_lock=Lock()),
            patch.object(runtime, "_configure_logging"),
            patch.object(runtime, "_configure_langfuse") as langfuse,
            patch.object(runtime, "_shutdown_langfuse"),
            patch.object(runtime, "_report_error"),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", side_effect=failure),
        ):
            with self.assertRaises(ValueError) as caught:
                runtime.configure_observability(ObservabilityConfig(otel=self.config))
            self.assertIs(caught.exception, failure)
            self.assertIs(runtime.running, False)
            langfuse.assert_not_called()
        self.assertTrue(self.metrics.closed)
        self.assertTrue(self.spans.closed)
        self.assertIsNone(otel._meter_provider)
        self.assertIsNone(otel._tracer_provider)

    def test_metric_shutdown_still_runs_after_trace_shutdown_failure(self):
        self.start()
        with patch.object(self.spans, "shutdown", side_effect=RuntimeError("shutdown failure")):
            with self.assertRaisesRegex(RuntimeError, "shutdown failure"):
                otel._shutdown_otel()
        self.assertTrue(self.metrics.closed)


if __name__ == "__main__":
    unittest.main()
