"""Request-scoped observability context shared by logs, OTel, and Langfuse."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Optional

from app.utils.id_generator import new_trace_id


@dataclass(frozen=True)
class ObservabilityContext:
    """Safe correlation metadata for one backend execution."""

    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    request_source: Optional[str] = None
    task_type: Optional[str] = None
    ctx: Optional[str] = None
    summary_id: Optional[str] = None
    graph_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    external_request_id: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    operation: Optional[str] = None
    llm_call_id: Optional[str] = None
    llm_call_count: int = 0


_CONTEXT: ContextVar[ObservabilityContext] = ContextVar(
    "ainsight_backend_context",
    default=ObservabilityContext(),
)


def get_observability_context() -> ObservabilityContext:
    return _CONTEXT.get()


def update_observability_context(**values: Any) -> ObservabilityContext:
    """Update selected request fields without passing a context object around."""

    current = get_observability_context()
    allowed = current.__dataclass_fields__
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown observability context fields: {sorted(unknown)}")
    updated = replace(current, **values)
    _CONTEXT.set(updated)
    return updated


@contextmanager
def observability_scope(**values: Any) -> Iterator[ObservabilityContext]:
    """Temporarily bind correlation fields and restore the parent scope."""

    current = get_observability_context()
    allowed = current.__dataclass_fields__
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown observability context fields: {sorted(unknown)}")
    token = _CONTEXT.set(replace(current, **values))
    try:
        yield get_observability_context()
    finally:
        _CONTEXT.reset(token)


def ensure_trace_id(preferred: Optional[str] = None) -> str:
    """Return the active OTel trace ID, scoped fallback, or a new W3C ID."""

    native_trace_id, _ = active_otel_ids()
    if native_trace_id:
        return native_trace_id
    current = get_observability_context().trace_id
    if current:
        return current
    if preferred and _is_w3c_trace_id(preferred):
        return preferred
    return new_trace_id()


@contextmanager
def trace_scope_if_missing() -> Iterator[str]:
    """Bind one temporary fallback trace across a direct non-request call."""

    native_trace_id, _ = active_otel_ids()
    current_trace_id = get_observability_context().trace_id
    if native_trace_id or current_trace_id:
        yield native_trace_id or current_trace_id  # type: ignore[misc]
        return
    trace_id = new_trace_id()
    with observability_scope(trace_id=trace_id):
        yield trace_id

def increment_llm_call_count() -> int:
    current = get_observability_context()
    ## parallel llm calls require thread safe increment
    count = current.llm_call_count + 1
    _CONTEXT.set(replace(current, llm_call_count=count))
    return count


def active_otel_ids() -> tuple[Optional[str], Optional[str]]:
    """Read native IDs lazily so local no-OTel execution remains lightweight."""

    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}", f"{span_context.span_id:016x}"
    except (ImportError, AttributeError):
        pass
    return None, None


def correlation_fields() -> dict[str, Any]:
    context = asdict(get_observability_context())
    native_trace_id, span_id = active_otel_ids()
    context["trace_id"] = native_trace_id or context.get("trace_id")
    context["span_id"] = span_id
    return {key: value for key, value in context.items() if value not in (None, "")}


def _is_w3c_trace_id(value: str) -> bool:
    return len(value) == 32 and value != "0" * 32 and all(character in "0123456789abcdef" for character in value)
