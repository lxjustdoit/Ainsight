"""Langfuse model observations, point events, and asynchronous SDK delivery."""

from dataclasses import dataclass, field
from _thread import LockType
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime
import logging
import time
from threading import Lock
from typing import Any, Iterator, Literal, Optional, Protocol

from opentelemetry.context import Context, attach, detach
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from app.core.config import LangfuseConfig
from app.services.observability.logging import log_error
from app.services.observability.otel import record_langfuse_drop
from app.services.observability.queue import (
    DirectObservabilityQueue,
    InMemoryObservabilityQueue,
    ObservabilityQueue,
)

ModelObservationType = Literal["generation", "embedding"]
ObservationInput = dict[str, str | int]
ObservationOutput = str | dict[str, int]


@dataclass(frozen=True)
class LangfuseObsStart:
    """Start data captured synchronously so Langfuse keeps real LLM timing."""

    trace_id: str
    llm_call_id: str
    task_type: str
    agent_name: str
    provider: str
    model_name: str
    prompt_version: str
    prompt: ObservationInput
    started_at_ns: int
    span_id: Optional[str] = None
    step_name: Optional[str] = None
    model_config_id: Optional[str] = None
    schema_version: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_type: ModelObservationType = "generation"


@dataclass(frozen=True)
class LangfuseObsEvent:
    """Completion-level record for one real LLM provider call."""

    trace_id: str
    llm_call_id: str
    task_type: str
    agent_name: str
    provider: str
    model_name: str
    prompt_version: str
    prompt: ObservationInput
    response: ObservationOutput
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: int
    status: str
    started_at_ns: int
    completed_at_ns: int
    usage_available: bool = True
    completion_start_time: Optional[datetime] = None
    span_id: Optional[str] = None
    step_name: Optional[str] = None
    model_config_id: Optional[str] = None
    schema_version: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False
    validation_result: Optional[str] = None
    time_to_first_event_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_type: ModelObservationType = "generation"


@dataclass(frozen=True)
class LangfusePointEvent:
    """TEST-ONLY capture shape for instantaneous SDK events."""

    llm_call_id: str
    name: str
    occurred_at_ns: int
    metadata: dict[str, Any]


@dataclass
class _ActiveObservation:
    """Route events and guard call completion; the sink owns the SDK observation."""

    llm_call_id: str
    completed_at_ns: Optional[int] = None
    lock: LockType = field(default_factory=Lock, repr=False)

    def finish(self) -> int:
        with self.lock:
            if self.completed_at_ns is None:
                self.completed_at_ns = time.time_ns()
            return self.completed_at_ns


_active_observation: ContextVar[Optional[_ActiveObservation]] = ContextVar("langfuse_active_observation", default=None)


class LangfuseEventSink(Protocol):
    def begin(self, event: LangfuseObsStart) -> None:
        ...

    def publish(self, event: LangfuseObsEvent) -> None:
        ...

    def record_event(self, llm_call_id: str, name: str, metadata: dict[str, Any]) -> bool:
        ...

    def discard(self, llm_call_id: str) -> None:
        ...

    def flush(self) -> None:
        ...

    def shutdown(self) -> None:
        ...


class NoopLangfuseEventSink:
    def begin(self, event: LangfuseObsStart) -> None:
        return None

    def publish(self, event: LangfuseObsEvent) -> None:
        return None

    def record_event(self, llm_call_id: str, name: str, metadata: dict[str, Any]) -> bool:
        return False

    def discard(self, llm_call_id: str) -> None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@dataclass
class InMemoryLangfuseEventSink:
    """TEST-ONLY deterministic observation capture without network export."""

    events: list[LangfuseObsEvent] = field(default_factory=list)
    starts: list[LangfuseObsStart] = field(default_factory=list)
    point_events: list[LangfusePointEvent] = field(default_factory=list)
    _active_calls: set[str] = field(default_factory=set, init=False)

    def begin(self, event: LangfuseObsStart) -> None:
        self.starts.append(event)
        self._active_calls.add(event.llm_call_id)

    def publish(self, event: LangfuseObsEvent) -> None:
        self.events.append(event)
        self._active_calls.discard(event.llm_call_id)

    def record_event(self, llm_call_id: str, name: str, metadata: dict[str, Any]) -> bool:
        if llm_call_id not in self._active_calls:
            return False
        self.point_events.append(LangfusePointEvent(llm_call_id, name, time.time_ns(), dict(metadata)))
        return True

    def discard(self, llm_call_id: str) -> None:
        self._active_calls.discard(llm_call_id)

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self._active_calls.clear()


class LangfuseSDKEventSink:
    """Adapter for the current Langfuse Python SDK observation API."""

    _claimed_public_keys: set[str] = set()
    _initialization_lock = Lock()

    def __init__(self, config: LangfuseConfig, *, service_name: str):
        if not config.public_key or not config.secret_key:
            raise ValueError("Langfuse public_key and secret_key are required when Langfuse is enabled")
        try:
            from langfuse import Langfuse
        except ImportError as exc:
            raise RuntimeError("The langfuse package is required when observability.langfuse.enabled is true") from exc
        # The SDK caches resources by public key even after shutdown. Rebuilding
        # with that key silently reuses a stopped processor; never alter SDK internals.
        with self._initialization_lock:
            if config.public_key in self._claimed_public_keys:
                raise RuntimeError("Langfuse was already initialized for this project; restart the process to reconfigure it")
            self._claimed_public_keys.add(config.public_key)
        # The SDK installs its own batch processor on this private provider.
        # Root IDs are shared with SigNoz without inventing an unexported parent.
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ALWAYS_ON,
            id_generator=_CorrelatedTraceIdGenerator(),
        )
        try:
            self._client = Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                base_url=config.host,
                debug=config.debug,
                tracing_enabled=True,
                flush_interval=max(0.1, config.flush_interval_ms / 1000.0),
                tracer_provider=self._provider,
            )
        except BaseException:
            try:
                self._provider.shutdown()
            except Exception:
                # Preserve the SDK construction exception, not the cleanup error.
                with suppress(Exception):
                    logging.getLogger("ainsight.observability.langfuse").exception("langfuse_startup_cleanup_failed")
            raise
        self._observations: dict[str, Any] = {}
        self._observations_lock = Lock()
        self._closed = False

    def begin(self, event: LangfuseObsStart) -> None:
        observation = self._start_observation(event)
        with self._observations_lock:
            previous = self._observations.pop(event.llm_call_id, None)
            self._observations[event.llm_call_id] = observation
        if previous is not None:
            previous.end()

    def _start_observation(self, event: LangfuseObsStart | LangfuseObsEvent) -> Any:
        trace_token = _langfuse_trace_id.set(int(event.trace_id, 16))
        context_token = attach(Context())
        try:
            return self._client.start_observation(
                name=event.step_name or event.task_type or "llm-call",
                as_type=event.observation_type,
                input=event.prompt,
                metadata=_start_metadata(event),
                version=event.prompt_version,
                model=event.model_name,
                model_parameters=_model_parameters(event.metadata),
            )
        finally:
            detach(context_token)
            _langfuse_trace_id.reset(trace_token)

    def record_event(self, llm_call_id: str, name: str, metadata: dict[str, Any]) -> bool:
        # Create a production SDK event now; the SDK exports it asynchronously.
        # Keep its parent alive while capturing the real occurrence time.
        with self._observations_lock:
            observation = self._observations.get(llm_call_id)
            if self._closed or observation is None:
                return False
            observation.create_event(name=name, metadata={**metadata, "llm_call_id": llm_call_id})
        return True

    def publish(self, event: LangfuseObsEvent) -> None:
        with self._observations_lock:
            observation = self._observations.pop(event.llm_call_id, None)
        timing_degraded = observation is None
        if observation is None:
            observation = self._start_observation(event)
        metadata = _completion_metadata(event)
        # Reconstructed observations cannot reproduce the original SDK start time.
        if timing_degraded:
            metadata["export_timing_degraded"] = True
        usage_fields = {}
        if event.usage_available:
            usage_fields["usage_details"] = {
                key: value for key, value in {
                    "input": event.input_tokens, "output": event.output_tokens, "total": event.total_tokens,
                }.items() if value is not None
            }
        try:
            observation.update(
                output=event.response,
                metadata=metadata,
                level="ERROR" if event.status == "error" else "DEFAULT",
                status_message=event.error_message,
                completion_start_time=event.completion_start_time,
                model_parameters=_model_parameters(event.metadata),
                **usage_fields,
            )
        finally:
            if timing_degraded:
                observation.end()
            else:
                observation.end(end_time=event.completed_at_ns)

    def discard(self, llm_call_id: str) -> None:
        with self._observations_lock:
            observation = self._observations.pop(llm_call_id, None)
        if observation is not None:
            observation.update(level="WARNING", status_message="Langfuse observation delivery was interrupted")
            observation.end()

    def flush(self) -> None:
        if not self._closed:
            self._client.flush()

    def shutdown(self) -> None:
        with self._observations_lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._observations.values())
            self._observations.clear()
        try:
            for observation in pending:
                observation.end()
            self._client.shutdown()
        finally:
            self._provider.shutdown()


_langfuse_trace_id: ContextVar[Optional[int]] = ContextVar("langfuse_root_trace_id", default=None)


class _CorrelatedTraceIdGenerator(RandomIdGenerator):
    """Reuse the request ID for independent Langfuse roots via the public OTel API."""

    def generate_trace_id(self) -> int:
        return _langfuse_trace_id.get() or super().generate_trace_id()


_runtime_lock = Lock()
_event_sink: LangfuseEventSink = NoopLangfuseEventSink()
_event_queue: ObservabilityQueue[LangfuseObsEvent] = DirectObservabilityQueue(_event_sink.publish)
_flush_timeout_seconds = 5.0
_capture_enabled = False


def _configure_langfuse(
    config: LangfuseConfig,
    *,
    sink: Optional[LangfuseEventSink] = None,
    synchronous: bool = False,
    service_name: str = "ainsight-backend",
) -> LangfuseEventSink:
    """Internal startup step called by runtime; not a reconfiguration API."""

    global _event_sink, _event_queue, _flush_timeout_seconds, _capture_enabled
    with _runtime_lock:
        _event_sink = sink or (LangfuseSDKEventSink(config, service_name=service_name) if config.enabled else NoopLangfuseEventSink())
        _capture_enabled = config.enabled or sink is not None
        _flush_timeout_seconds = float(config.queue.flush_timeout_seconds)
        if synchronous or not config.enabled:
            _event_queue = DirectObservabilityQueue(_event_sink.publish)
        else:
            _event_queue = InMemoryObservabilityQueue(
                _event_sink.publish,
                max_size=config.queue.max_size,
                enqueue_timeout_ms=config.queue.enqueue_timeout_ms,
                drop_when_full=config.queue.drop_when_full,
                name="ainsight-langfuse-exporter",
                on_drop=_handle_dropped_event,
            )
        return _event_sink


def emit_langfuse_event(event: LangfuseObsEvent) -> bool:
    return _event_queue.enqueue(event)


@contextmanager
def langfuse_observation_scope(event: LangfuseObsStart) -> Iterator[None]:
    """Bind a real model observation for custom events until its call returns."""
    active = _ActiveObservation(event.llm_call_id) if begin_langfuse_observation(event) else None
    token = _active_observation.set(active)
    try:
        yield
    finally:
        if active is not None:
            active.finish()
        _active_observation.reset(token)


def finish_langfuse_event_scope() -> int:
    """Stop accepting events before the completion payload captures its end time."""
    active = _active_observation.get()
    return active.finish() if active is not None else time.time_ns()


def record_llm_event(name: str, **metadata: Any) -> bool:
    """Record one instantaneous child of the active model call, best effort."""
    active = _active_observation.get()
    if not _capture_enabled or active is None:
        return False
    try:
        with active.lock:
            if active.completed_at_ns is not None:
                return False
            return _event_sink.record_event(active.llm_call_id, name, metadata)
    except Exception as exc:
        # A failed event must neither terminate its generation nor fail the call.
        with suppress(Exception):
            log_error("langfuse_event_record_failed", exc, dependency="langfuse")
        return False


def _handle_dropped_event(event: LangfuseObsEvent, reason: str) -> None:
    record_langfuse_drop(reason)
    discard_langfuse_observation(event.llm_call_id)


def discard_langfuse_observation(llm_call_id: str) -> None:
    try:
        _event_sink.discard(llm_call_id)
    except Exception:
        logging.getLogger("ainsight.observability.langfuse").exception("langfuse_observation_discard_failed")


def begin_langfuse_observation(event: LangfuseObsStart) -> bool:
    if not _capture_enabled:
        return False
    try:
        _event_sink.begin(event)
        return True
    except Exception:
        logging.getLogger("ainsight.observability.langfuse").exception("langfuse_observation_start_failed")
        return False


def langfuse_capture_enabled() -> bool:
    """Return whether prompt/response events have an active destination."""

    return _capture_enabled


def flush_langfuse() -> bool:
    """Drain application events and flush SDK batches for diagnostics or graceful exit."""
    drained = _event_queue.flush(_flush_timeout_seconds)
    _event_sink.flush()
    return drained


def _shutdown_langfuse() -> None:
    global _capture_enabled, _event_sink, _event_queue
    with _runtime_lock:
        _capture_enabled = False
        if _shutdown_locked():
            _event_sink = NoopLangfuseEventSink()
            _event_queue = DirectObservabilityQueue(_event_sink.publish)


def _shutdown_locked() -> bool:
    if not _event_queue.shutdown(_flush_timeout_seconds):
        logging.getLogger("ainsight.observability.langfuse").warning("langfuse_queue_shutdown_timed_out")
        return False
    # SDK shutdown already flushes; a separate flush failure must not skip close.
    _event_sink.shutdown()
    return True


def _start_metadata(event: LangfuseObsStart | LangfuseObsEvent) -> dict[str, Any]:
    metadata = {
        "llm_call_id": event.llm_call_id,
        "task_type": event.task_type,
        "agent_name": event.agent_name,
        "provider": event.provider,
        "model_config_id": event.model_config_id,
        "prompt_version": event.prompt_version,
        "schema_version": event.schema_version,
        "step_name": event.step_name,
        "retry_count": event.retry_count,
        "fallback_used": event.fallback_used,
        "otel_span_id": event.span_id,
        "started_at_ns": event.started_at_ns,
        **event.metadata,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _completion_metadata(event: LangfuseObsEvent) -> dict[str, Any]:
    metadata = {
        **_start_metadata(event),
        "status": event.status,
        "validation_result": event.validation_result,
        "latency_ms": event.latency_ms,
        "time_to_first_event_ms": event.time_to_first_event_ms,
        "error_type": event.error_type,
        "error_message": event.error_message,
        "completed_at_ns": event.completed_at_ns,
        "usage_available": event.usage_available,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _model_parameters(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "temperature": metadata.get("temperature"),
            "max_tokens": metadata.get("max_output_tokens"),
        }.items()
        if value is not None
    }
