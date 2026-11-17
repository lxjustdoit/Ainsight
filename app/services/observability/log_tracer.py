"""Business-operation decorator that combines concise logs with OTel spans."""

import asyncio
from contextlib import aclosing
import functools
import inspect
import time
from typing import Any, AsyncIterator, Callable, Iterator, Optional, TypeVar

from app.core.errors import AinsightError
from app.services.observability.context import (
    ensure_trace_id,
    get_observability_context,
    observability_scope,
    update_observability_context,
)
from app.services.observability.logging import log_error, log_info
from app.services.observability.otel import record_span_error, set_span_attributes, set_span_status, start_span


FunctionT = TypeVar("FunctionT", bound=Callable[..., Any])


def log_tracer(
    operation: str,
    *,
    task_type: Optional[str] = None,
    path: Optional[str] = None,
    method: str = "POST",
) -> Callable[[FunctionT], FunctionT]:
    """Trace a main business operation without passing recorder objects."""

    def decorate(function: FunctionT) -> FunctionT:
        if inspect.isasyncgenfunction(function):

            @functools.wraps(function)
            async def async_generator_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                async with aclosing(_run_async_generator(function, args, kwargs, operation, task_type, path, method)) as stream:
                    async for item in stream:
                        yield item

            return async_generator_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _operation_scope(function, args, kwargs, operation, task_type, path, method) as started_at:
                    try:
                        result = await function(*args, **kwargs)
                    except asyncio.CancelledError:
                        _record_operation_finished(operation, started_at, status="cancelled")
                        raise
                    except Exception as exc:
                        _record_operation_error(function, operation, started_at, exc)
                        raise
                    _record_operation_success(operation, started_at)
                    return result

            return async_wrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(function):

            @functools.wraps(function)
            def generator_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
                yield from _run_generator(function, args, kwargs, operation, task_type, path, method)

            return generator_wrapper  # type: ignore[return-value]

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _operation_scope(function, args, kwargs, operation, task_type, path, method) as started_at:
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    _record_operation_error(function, operation, started_at, exc)
                    raise
                _record_operation_success(operation, started_at)
                return result

        return wrapper  # type: ignore[return-value]

    return decorate


def _run_generator(
    function: Callable[..., Iterator[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    operation: str,
    task_type: Optional[str],
    path: Optional[str],
    method: str,
) -> Iterator[Any]:
    with _operation_scope(function, args, kwargs, operation, task_type, path, method) as started_at:
        handled_failure = False
        try:
            for item in function(*args, **kwargs):
                if getattr(item, "event", None) == "error":
                    handled_failure = True
                yield item
        except GeneratorExit:
            _record_operation_finished(operation, started_at, status="cancelled")
            raise
        except Exception as exc:
            _record_operation_error(function, operation, started_at, exc)
            raise
        if handled_failure:
            _record_operation_finished(operation, started_at, status="failed")
        else:
            _record_operation_success(operation, started_at)


async def _run_async_generator(
    function: Callable[..., AsyncIterator[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    operation: str,
    task_type: Optional[str],
    path: Optional[str],
    method: str,
) -> AsyncIterator[Any]:
    with _operation_scope(function, args, kwargs, operation, task_type, path, method) as started_at:
        handled_failure = False
        try:
            # Close in the consuming task so nested spans/contextvars unwind here,
            # not later in an unrelated async-generator finalizer task.
            async with aclosing(function(*args, **kwargs)) as stream:
                async for item in stream:
                    if getattr(item, "event", None) == "error":
                        handled_failure = True
                    yield item
        except (GeneratorExit, asyncio.CancelledError):
            _record_operation_finished(operation, started_at, status="cancelled")
            raise
        except Exception as exc:
            _record_operation_error(function, operation, started_at, exc)
            raise
        if handled_failure:
            _record_operation_finished(operation, started_at, status="failed")
        else:
            _record_operation_success(operation, started_at)


class _operation_scope:
    def __init__(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        operation: str,
        task_type: Optional[str],
        path: Optional[str],
        method: str,
    ):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.operation = operation
        self.task_type = task_type
        self.path = path
        self.method = method
        self._span_scope: Any = None
        self._context_scope: Any = None
        self._parent_context: Any = None

    def __enter__(self) -> float:
        values = _bound_context_values(self.function, self.args, self.kwargs)
        current = get_observability_context()
        self._parent_context = current
        for field_name in (
            "user_id",
            "request_source",
            "ctx",
            "summary_id",
            "graph_id",
            "conversation_id",
            "request_id",
            "external_request_id",
        ):
            if values.get(field_name) is None:
                values[field_name] = getattr(current, field_name)
        values.update(
            {
                "trace_id": ensure_trace_id(),
                "operation": self.operation,
                "task_type": self.task_type or values.get("task_type"),
                "path": self.path or values.get("path"),
                "method": self.method,
                "llm_call_count": 0,
            }
        )
        span_attributes = _span_attributes(values)
        self._span_scope = start_span(self.operation, span_attributes)
        self._span_scope.__enter__()
        native_trace_id = ensure_trace_id(values["trace_id"])
        values["trace_id"] = native_trace_id
        self._context_scope = observability_scope(**{key: value for key, value in values.items() if value is not None})
        self._context_scope.__enter__()
        log_info("operation_started", operation=self.operation)
        return time.perf_counter()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        completed_context = get_observability_context()
        self._context_scope.__exit__(exc_type, exc, traceback)
        if self._parent_context.trace_id:
            propagated = {
                field_name: getattr(completed_context, field_name)
                for field_name in (
                    "task_type",
                    "ctx",
                    "summary_id",
                    "graph_id",
                    "conversation_id",
                )
                if getattr(completed_context, field_name) is not None
            }
            propagated["llm_call_count"] = (
                self._parent_context.llm_call_count + completed_context.llm_call_count
            )
            update_observability_context(**propagated)
        self._span_scope.__exit__(exc_type, exc, traceback)


def _record_operation_success(operation: str, started_at: float) -> None:
    _record_operation_finished(operation, started_at, status="success")


def _record_operation_finished(operation: str, started_at: float, *, status: str) -> None:
    context = get_observability_context()
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    set_span_attributes(
        **{
            "ainsight.status": status,
            "ainsight.llm_call_count": context.llm_call_count,
            **_span_attributes(vars(context)),
        }
    )
    set_span_status(status)
    log_info(
        "operation_finished",
        status=status,
        operation=operation,
        duration_ms=duration_ms,
        llm_call_count=context.llm_call_count,
    )


def _record_operation_error(
    function: Callable[..., Any],
    operation: str,
    started_at: float,
    error: Exception,
) -> None:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    error_code = error.error_code if isinstance(error, AinsightError) else "INTERNAL_ERROR"
    fields = {
        "operation": operation,
        "duration_ms": duration_ms,
        "error_code": error_code,
        "error_location": f"{function.__module__}.{function.__qualname__}",
    }
    record_span_error("business_error", error, **fields)
    log_error("business_error", error, **fields)
    try:
        setattr(error, "_ainsight_business_error_logged", True)
    except (AttributeError, TypeError):
        pass


def _bound_context_values(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        arguments = inspect.signature(function).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        arguments = kwargs
    return {
        "request_source": arguments.get("request_source"),
        "ctx": arguments.get("context_id") or arguments.get("ctx"),
        "summary_id": arguments.get("summary_id"),
        "graph_id": arguments.get("graph_id"),
        "conversation_id": arguments.get("conversation_id"),
        "request_id": arguments.get("request_id"),
        "user_id": arguments.get("user_id"),
        "external_request_id": arguments.get("external_request_id"),
    }


def _span_attributes(values: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "task_type": "ainsight.task_type",
        "user_id": "ainsight.user_id",
        "request_source": "ainsight.request_source",
        "ctx": "ainsight.ctx",
        "summary_id": "ainsight.summary_id",
        "graph_id": "ainsight.graph_id",
        "conversation_id": "ainsight.conversation_id",
        "request_id": "ainsight.request_id",
        "external_request_id": "ainsight.external_request_id",
        "operation": "ainsight.operation",
    }
    return {attribute: values[key] for key, attribute in mapping.items() if values.get(key) is not None}
