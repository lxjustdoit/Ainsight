"""OpenTelemetry setup, ASGI request tracing, metrics, and manual span helpers."""

from contextlib import nullcontext, suppress
import time
import traceback
from typing import Any, Callable, Optional

from app.core.config import OTelConfig
from app.utils.id_generator import new_trace_id
from app.services.observability.context import active_otel_ids, get_observability_context, observability_scope
from app.services.observability.logging import (
    log_error,
    log_info,
    redact_sensitive_text,
    sanitize_telemetry_fields,
)


_tracer: Any = None
_meter: Any = None
_langfuse_dropped: Any = None
_request_duration: Any = None
_tracer_provider: Any = None
_meter_provider: Any = None
_otel_tracing_enabled = False
_otel_metrics_enabled = False


def _configure_otel(config: OTelConfig, *, service_name: str) -> None:
    """Internal startup step for bounded async OTLP export when enabled."""

    global _tracer, _meter, _langfuse_dropped, _request_duration, _tracer_provider, _meter_provider
    global _otel_tracing_enabled, _otel_metrics_enabled

    try:
        from opentelemetry import metrics, trace
    except ImportError:
        if config.enabled:
            raise RuntimeError("OpenTelemetry packages are required when observability.otel.enabled is true")
        return

    _otel_tracing_enabled = config.enabled and config.traces.enabled
    _otel_metrics_enabled = config.enabled and config.metrics.enabled

    # SDK processor instruments bind to their meter when the processor is created.
    # Build this private provider first; do not rely on a global MeterProvider.
    if _otel_metrics_enabled and _meter_provider is None:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        unowned_resource = None
        try:
            unowned_resource = _build_metric_exporter(config)
            unowned_resource = PeriodicExportingMetricReader(unowned_resource)
            _meter_provider = MeterProvider(
                resource=Resource.create({"service.name": service_name}),
                metric_readers=[unowned_resource],
            )
        except BaseException:
            _cleanup_unowned_resource(unowned_resource)
            raise

    if _otel_tracing_enabled and _tracer_provider is None:
        from opentelemetry.metrics import NoOpMeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        class ScopedIdGenerator(RandomIdGenerator):
            def generate_trace_id(self) -> int:
                scoped_id = get_observability_context().trace_id
                return int(scoped_id, 16) if scoped_id else super().generate_trace_id()

        _tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(_sampling_ratio(config.traces.sampling_ratio))),
            id_generator=ScopedIdGenerator(),
        )
        unowned_resource = None
        try:
            unowned_resource = _build_trace_exporter(config)
            unowned_resource = BatchSpanProcessor(
                unowned_resource,
                max_queue_size=config.queue.max_queue_size,
                schedule_delay_millis=config.queue.schedule_delay_ms,
                max_export_batch_size=config.queue.max_export_batch_size,
                export_timeout_millis=config.queue.export_timeout_ms,
                # The SDK environment flag controls internal instruments. When
                # app metrics are disabled, prevent fallback to a global provider.
                meter_provider=_meter_provider if _otel_metrics_enabled else NoOpMeterProvider(),
            )
            _tracer_provider.add_span_processor(unowned_resource)
        except BaseException:
            _cleanup_unowned_resource(unowned_resource)
            raise

    _tracer = _tracer_provider.get_tracer("ainsight.observability") if _otel_tracing_enabled else None

    if _otel_metrics_enabled:
        _meter = _meter_provider.get_meter("ainsight.observability")
        _langfuse_dropped = _meter.create_counter("ainsight.langfuse.queue.dropped", unit="{event}")
        _request_duration = _meter.create_histogram(
            "http.server.request.duration", unit="s",
            explicit_bucket_boundaries_advisory=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10),
        )
    else:
        _meter = None
        _langfuse_dropped = None
        _request_duration = None


def _cleanup_unowned_resource(resource: Any) -> None:
    """Clean up local startup allocations not yet owned by a provider."""
    if resource is None:
        return
    try:
        resource.shutdown()
    except Exception as exc:
        # Retain the original initialization error even if cleanup/reporting fails.
        with suppress(Exception):
            log_error("observability_shutdown_failed", exc, dependency="otel_startup")


def start_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    *,
    kind: Any = None,
    context: Any = None,
) -> Any:
    if not _otel_tracing_enabled:
        return nullcontext(_NoopSpan())
    tracer = _tracer
    if tracer is None:
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("ainsight.backend")
        except ImportError:
            return nullcontext(_NoopSpan())
    options: dict[str, Any] = {
        "attributes": _otel_attributes(attributes or {}),
        # Exception telemetry is emitted once by record_span_error. Content redaction is temporarily paused.
        "record_exception": False,
        "set_status_on_exception": False,
    }
    if kind is not None:
        options["kind"] = kind
    if context is not None:
        options["context"] = context
    return tracer.start_as_current_span(name, **options)


def record_span_event(event: str, **fields: Any) -> None:
    if not _otel_tracing_enabled:
        return
    span = _current_span()
    if span is not None and span.is_recording():
        span.add_event(event, attributes=_otel_attributes(fields))


def record_span_error(event: str, error: BaseException, **fields: Any) -> None:
    if not _otel_tracing_enabled:
        return
    span = _current_span()
    if span is None or not span.is_recording():
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        error_message = redact_sensitive_text(error, max_length=2000)
        stacktrace = redact_sensitive_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            max_length=10000,
        )
        span.add_event(
            "exception",
            attributes=_otel_attributes(
                {
                    "exception.type": type(error).__name__,
                    "exception.message": error_message,
                    "exception.stacktrace": stacktrace,
                    "event.name": event,
                    **fields,
                }
            ),
        )
        span.set_status(Status(StatusCode.ERROR, error_message))
        span.set_attribute("error.type", type(error).__name__)
        span.set_attribute("error.message", error_message)
    except ImportError:
        return


def set_span_attributes(**fields: Any) -> None:
    if not _otel_tracing_enabled:
        return
    span = _current_span()
    if span is None or not span.is_recording():
        return
    for key, value in _otel_attributes(fields).items():
        span.set_attribute(key, value)


def _shutdown_otel() -> None:
    global _meter_provider, _tracer_provider, _meter, _tracer, _request_duration, _langfuse_dropped
    global _otel_tracing_enabled, _otel_metrics_enabled
    # Draining spans records final processor counts; collect metrics afterwards.
    resources = (_tracer_provider, _meter_provider)
    _meter_provider = _tracer_provider = _meter = _tracer = None
    _request_duration = _langfuse_dropped = None
    _otel_tracing_enabled = _otel_metrics_enabled = False
    first_error = None
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.shutdown()
        except Exception as exc:
            first_error = first_error or exc
    if first_error is not None:
        raise first_error


def record_langfuse_drop(reason: str) -> None:
    """Count terminal losses in the application queue, even without an active span."""
    if _langfuse_dropped is not None:
        _langfuse_dropped.add(1, {"reason": reason})
    record_span_event("langfuse.queue.dropped", reason=reason)


def set_span_status(status: str, error_type: Optional[str] = None) -> None:
    """Set execution outcome independently of detailed exception events."""
    span = _current_span() if _otel_tracing_enabled else None
    if span is None or not span.is_recording():
        return
    from opentelemetry.trace import Status, StatusCode

    span.set_attribute("ainsight.status", status)
    if status in {"error", "failed"}:
        span.set_status(Status(StatusCode.ERROR))
        span.set_attribute("error.type", error_type or "operation_failed")


class ObservabilityMiddleware:
    """Framework-neutral ASGI middleware for request trace and log correlation."""

    def __init__(self, app: Callable[..., Any]):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        headers = _headers(scope)
        status_code: Optional[int] = None
        error_type: Optional[str] = None
        scheme = str(scope.get("scheme", "http"))
        started_at = time.perf_counter()
        preferred_trace_id = new_trace_id()

        try:
            from opentelemetry.trace import SpanKind

            kind = SpanKind.SERVER
        except ImportError:
            kind = None

        with observability_scope(trace_id=preferred_trace_id), start_span(
            f"HTTP {method}",
            {"http.request.method": method, "url.path": path, "url.scheme": scheme},
            kind=kind,
            context=_new_root_context(),
        ):
            trace_id, _ = active_otel_ids()
            trace_id = trace_id or preferred_trace_id
            with observability_scope(
                trace_id=trace_id,
                method=method,
                path=path,
                request_source=headers.get("x-request-source"),
                request_id=_idempotency_key(headers.get("idempotency-key")),
                external_request_id=headers.get("x-request-id"),
                llm_call_count=0,
            ):

                async def send_with_trace(message: dict[str, Any]) -> None:
                    nonlocal status_code
                    if message.get("type") == "http.response.start":
                        status_code = int(message.get("status", 500))
                        response_headers = [
                            (key, value)
                            for key, value in message.get("headers", [])
                            if key.lower() != b"x-trace-id"
                        ]
                        response_headers.append((b"x-trace-id", trace_id.encode("ascii")))
                        message = {**message, "headers": response_headers}
                    await send(message)

                try:
                    await self.app(scope, receive, send_with_trace)
                except Exception as exc:
                    error_type = type(exc).__name__
                    if not getattr(exc, "_ainsight_business_error_logged", False):
                        record_span_error("request.error", exc)
                        log_error(
                            "business_error",
                            exc,
                            status_code=status_code,
                            error_location="asgi.request",
                        )
                    raise
                finally:
                    duration_seconds = time.perf_counter() - started_at
                    route_path = _route_path(scope)
                    _record_http_span_status(status_code, method=method, route_path=route_path, error_type=error_type)
                    log_info(
                        "request_finished",
                        status="success" if error_type is None and status_code is not None and status_code < 400 else "failed",
                        status_code=status_code,
                        duration_ms=int(duration_seconds * 1000),
                    )
                    _record_request_metrics(method, route_path, status_code, duration_seconds, scheme, error_type)


def _record_request_metrics(
    method: str, path: Optional[str], status_code: Optional[int], duration_seconds: float,
    scheme: str, error_type: Optional[str] = None,
) -> None:
    if _request_duration is None:
        return
    attributes: dict[str, Any] = {"http.request.method": method, "url.scheme": scheme}
    if path:
        attributes["http.route"] = path
    if status_code is not None:
        attributes["http.response.status_code"] = status_code
    if error_type or (status_code is not None and status_code >= 500):
        attributes["error.type"] = error_type or str(status_code)
    _request_duration.record(duration_seconds, attributes)


def _record_http_span_status(
    status_code: Optional[int],
    *,
    method: Optional[str] = None,
    route_path: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    if not _otel_tracing_enabled:
        return
    span = _current_span()
    if span is None or not span.is_recording():
        return
    if status_code is not None:
        span.set_attribute("http.response.status_code", status_code)
    if route_path:
        span.set_attribute("http.route", route_path)
        if method:
            span.update_name(f"{method} {route_path}")
    # HTTP server 4xx are client errors; dashboards can still count them by code.
    # Log deduplication must never suppress the parent request's failure state.
    if error_type or (status_code is not None and status_code >= 500):
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, error_type or f"HTTP {status_code}"))
        span.set_attribute("error.type", error_type or str(status_code))


def _build_trace_exporter(config: OTelConfig) -> Any:
    if config.exporter_protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError("The OTLP gRPC trace exporter is not installed") from exc
        return OTLPSpanExporter(
            endpoint=config.exporter_endpoint,
            headers=config.exporter_headers,
            insecure=config.exporter_insecure,
        )
    if config.exporter_protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=_http_endpoint(config.exporter_endpoint, "traces"), headers=config.exporter_headers)
    raise ValueError(f"Unsupported OTLP protocol: {config.exporter_protocol}")


def _build_metric_exporter(config: OTelConfig) -> Any:
    if config.exporter_protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        except ImportError as exc:
            raise RuntimeError("The OTLP gRPC metric exporter is not installed") from exc
        return OTLPMetricExporter(
            endpoint=config.exporter_endpoint,
            headers=config.exporter_headers,
            insecure=config.exporter_insecure,
        )
    if config.exporter_protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter(endpoint=_http_endpoint(config.exporter_endpoint, "metrics"), headers=config.exporter_headers)
    raise ValueError(f"Unsupported OTLP protocol: {config.exporter_protocol}")


def _http_endpoint(endpoint: str, signal: str) -> str:
    value = endpoint.rstrip("/")
    return value if value.endswith(f"/v1/{signal}") else f"{value}/v1/{signal}"


def _sampling_ratio(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("OTel trace sampling_ratio must be between 0.0 and 1.0")
    return value


def _current_span() -> Any:
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except ImportError:
        return None


def _otel_attributes(fields: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, value in sanitize_telemetry_fields(fields).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            attributes[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
            attributes[str(key)] = list(value)
        else:
            attributes[str(key)] = str(value)
    return attributes


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def _idempotency_key(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("i_"):
        return value
    return None


def _new_root_context() -> Any:
    try:
        from opentelemetry.context import Context

        return Context()
    except ImportError:
        return None


def _route_path(scope: dict[str, Any]) -> Optional[str]:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    return str(route_path) if route_path else None


class _NoopSpan:
    def is_recording(self) -> bool:
        return False
