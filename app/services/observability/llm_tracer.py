"""Shared model-call lifecycle for logs, OTel, and Langfuse observations."""

import functools
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, TypeVar

from pydantic import BaseModel

from app.utils.id_generator import new_llm_call_id
from app.services.observability.context import (
    active_otel_ids,
    ensure_trace_id,
    increment_llm_call_count,
    observability_scope,
)
from app.services.observability.langfuse_client import (
    LangfuseObsEvent,
    LangfuseObsStart,
    ModelObservationType,
    ObservationInput,
    ObservationOutput,
    emit_langfuse_event,
    finish_langfuse_event_scope,
    langfuse_capture_enabled,
    langfuse_observation_scope,
)
from app.services.observability.logging import log_error, log_info, redact_sensitive_text
from app.services.observability.otel import start_span, set_span_status, set_span_attributes


FunctionT = TypeVar("FunctionT", bound=Callable[..., Any])


def llm_tracer(
    step_name: Optional[str] = None,
    *,
    observation_type: ModelObservationType = "generation",
) -> Callable[[FunctionT], FunctionT]:
    """Record each decorated provider invocation as one distinct LLM call."""

    def decorate(function: FunctionT) -> FunctionT:
        if inspect.isgeneratorfunction(function):

            @functools.wraps(function)
            def generator_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
                yield from _trace_generator(function, args, kwargs, step_name, observation_type)

            return generator_wrapper  # type: ignore[return-value]

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = _LLMCall(function, args, kwargs, step_name, observation_type)
            with call.scope():
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    call.finish_error(exc)
                    raise
                call.finish_success(result)
                return result

        return wrapper  # type: ignore[return-value]

    return decorate

## trace for streaming LLM calls
def _trace_generator(
    function: Callable[..., Iterator[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    step_name: Optional[str],
    observation_type: ModelObservationType,
) -> Iterator[Any]:
    call = _LLMCall(function, args, kwargs, step_name, observation_type)
    chunks: list[str] = []
    final_result: Any = None
    with call.scope():
        try:
            for item in function(*args, **kwargs):
                event = getattr(item, "event", None)
                if event == "delta":
                    chunks.append(str(getattr(item, "content", "")))
                    call.mark_first_event()
                elif event == "final":
                    final_result = item
                yield item
        except GeneratorExit:
            call.finish_cancelled("".join(chunks))
            raise
        except Exception as exc:
            call.finish_error(exc, response="".join(chunks))
            raise
        call.finish_success(final_result, response="".join(chunks))


class _LLMCall:
    def __init__(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        step_name: Optional[str],
        observation_type: ModelObservationType,
    ):
        self.arguments = _bound_arguments(function, args, kwargs)
        self.model_config = self.arguments.get("model_config")
        self.observation_type = observation_type
        self.step_name = step_name or _value(self.model_config, "task_type", function.__name__)
        self.llm_call_id = new_llm_call_id()
        self.trace_id = ensure_trace_id()
        _, self.span_id = active_otel_ids()
        self.started_at = time.perf_counter()
        self.started_at_ns = time.time_ns()
        self.first_event_at: Optional[float] = None
        self.stream_event_count = 0
        self.attempt = int(self.arguments.get("attempt", 0) or 0)

    @contextmanager
    def scope(self) -> Iterator[None]:
        from opentelemetry.trace import SpanKind

        increment_llm_call_count()
        attributes = {
            "gen_ai.operation.name": self.operation_name,
            "gen_ai.provider.name": self.provider,
            "gen_ai.request.model": self.model_name,
            "ainsight.task_type": self.task_type,
            "ainsight.llm_call_id": self.llm_call_id,
        }
        with start_span(f"{self.operation_name} {self.model_name}", attributes, kind=SpanKind.CLIENT):
            self.trace_id = ensure_trace_id(self.trace_id)
            _, self.span_id = active_otel_ids()
            with observability_scope(
                trace_id=self.trace_id, llm_call_id=self.llm_call_id, task_type=self.task_type,
            ):
                log_info("llm_call_started", step_name=self.step_name, provider=self.provider, model_name=self.model_name)
                with langfuse_observation_scope(self._start_event()):
                    yield

    def mark_first_event(self) -> None:
        self.stream_event_count += 1
        if self.first_event_at is None:
            self.first_event_at = time.perf_counter()

    def finish_success(self, result: Any, *, response: Optional[str] = None) -> None:
        usage = _extract_usage(result)
        if self.observation_type == "embedding":
            vectors = result.vectors
            output: ObservationOutput = {"vector_count": len(vectors), "dimensions": len(vectors[0]) if vectors else 0}
        else:
            output = response if response is not None else _extract_output(result)
        set_span_status("success")
        self._emit("success", output, usage, validation_result=_validation_result_from_result(result))
        log_info(
            "llm_call_finished",
            task_type=self.task_type,
            step_name=self.step_name,
            provider=self.provider,
            model_name=self.model_name,
            duration_ms=self.duration_ms,
            output_chars=len(output) if isinstance(output, str) else None,
            stream_event_count=self.stream_event_count,
        )

    def finish_error(self, error: Exception, *, response: str = "") -> None:
        set_span_status("failed", type(error).__name__)
        # Full provider payloads belong in Langfuse; the client span identifies the failure.
        set_span_attributes(**{"ainsight.error_location": self.error_location})
        provider_result = getattr(error, "_ainsight_provider_result", None)
        failure_response = str(getattr(error, "_ainsight_raw_response", response))
        self._emit(
            "error",
            failure_response,
            _extract_usage(provider_result),
            error=error,
            validation_result=_validation_result(error),
        )
        log_error(
            "dependency_error",
            error,
            dependency="llm_provider",
            task_type=self.task_type,
            step_name=self.step_name,
            provider=self.provider,
            model_name=self.model_name,
            duration_ms=self.duration_ms,
            error_location=self.error_location,
        )

    def finish_cancelled(self, response: str) -> None:
        set_span_status("cancelled")
        self._emit("cancelled", response, _Usage(), validation_result="cancelled")

    def _emit(
        self,
        status: str,
        response: ObservationOutput,
        usage: "_Usage",
        *,
        error: Optional[Exception] = None,
        validation_result: Optional[str] = None,
    ) -> None:
        completed_at_ns = finish_langfuse_event_scope()
        if not langfuse_capture_enabled():
            return
        event = LangfuseObsEvent(
            trace_id=self.trace_id,
            span_id=self.span_id,
            llm_call_id=self.llm_call_id,
            task_type=self.task_type,
            agent_name=_agent_name(self.task_type),
            provider=self.provider,
            model_name=self.model_name,
            model_config_id=_value(self.model_config, "model_config_id", "") or None,
            prompt_version=_value(self.model_config, "prompt_version", "") or "unknown",
            schema_version=_schema_version(
                self.arguments.get("schema_output_type")
                or self.arguments.get("validation_output_type")
                or self.arguments.get("output_type")
            ),
            step_name=self.step_name,
            prompt=self.observation_input,
            response=response,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            usage_available=usage.available,
            latency_ms=self.duration_ms,
            started_at_ns=self.started_at_ns,
            completed_at_ns=completed_at_ns,
            completion_start_time=self.completion_start_time,
            time_to_first_event_ms=self.time_to_first_event_ms,
            retry_count=self.attempt,
            fallback_used="fallback" in self.task_type,
            status=status,
            validation_result=validation_result,
            error_type=type(error).__name__ if error else None,
            error_message=redact_sensitive_text(error, max_length=2000) if error else None,
            metadata={
                "temperature": _value(self.model_config, "temperature", None),
                "max_output_tokens": _value(self.model_config, "max_output_tokens", None),
                "output_chars": len(response) if isinstance(response, str) else None,
                "stream_event_count": self.stream_event_count,
            },
            observation_type=self.observation_type,
        )
        emit_langfuse_event(event)

    def _start_event(self) -> LangfuseObsStart:
        return LangfuseObsStart(
            trace_id=self.trace_id,
            span_id=self.span_id,
            llm_call_id=self.llm_call_id,
            task_type=self.task_type,
            agent_name=_agent_name(self.task_type),
            provider=self.provider,
            model_name=self.model_name,
            model_config_id=_value(self.model_config, "model_config_id", "") or None,
            prompt_version=_value(self.model_config, "prompt_version", "") or "unknown",
            schema_version=_schema_version(
                self.arguments.get("schema_output_type")
                or self.arguments.get("validation_output_type")
                or self.arguments.get("output_type")
            ),
            step_name=self.step_name,
            prompt=self.observation_input,
            started_at_ns=self.started_at_ns,
            retry_count=self.attempt,
            fallback_used="fallback" in self.task_type,
            metadata={
                "temperature": _value(self.model_config, "temperature", None),
                "max_output_tokens": _value(self.model_config, "max_output_tokens", None),
            },
            observation_type=self.observation_type,
        )

    @property
    def observation_input(self) -> ObservationInput:
        if self.observation_type == "embedding":
            texts = self.arguments.get("texts", ())
            return {"input_count": len(texts), "input_chars": sum(len(text) for text in texts)}
        return {
            "system": str(self.arguments.get("system_prompt", "")),
            "user": str(self.arguments.get("user_prompt", "")),
        }

    @property
    def operation_name(self) -> str:
        return "embeddings" if self.observation_type == "embedding" else "chat"

    @property
    def error_location(self) -> str:
        return "embeddings.provider_call" if self.observation_type == "embedding" else "pydantic_ai.provider_call"

    @property
    def task_type(self) -> str:
        default = "embedding" if self.observation_type == "embedding" else "unknown"
        return str(_value(self.model_config, "task_type", default))

    @property
    def provider(self) -> str:
        return str(_value(self.model_config, "provider", "unknown"))

    @property
    def model_name(self) -> str:
        return str(_value(self.model_config, "model_name", "unknown"))

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    @property
    def time_to_first_event_ms(self) -> Optional[int]:
        if self.first_event_at is None:
            return None
        return int((self.first_event_at - self.started_at) * 1000)

    @property
    def completion_start_time(self) -> Optional[datetime]:
        if self.first_event_at is None:
            return None
        first_event_ns = self.started_at_ns + int((self.first_event_at - self.started_at) * 1_000_000_000)
        return datetime.fromtimestamp(first_event_ns / 1_000_000_000, tz=timezone.utc)


def _bound_arguments(function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(inspect.signature(function).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _extract_output(result: Any) -> str:
    raw_output = getattr(result, "_ainsight_raw_output", None)
    if raw_output is not None:
        return _extract_output(raw_output)
    output = getattr(result, "output", result)
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except TypeError:
        return str(output)


@dataclass(frozen=True)
class _Usage:
    """Optional provider usage; missing counters are never manufactured as zero."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def available(self) -> bool:
        return any(value is not None for value in (self.input_tokens, self.output_tokens, self.total_tokens))


def _extract_usage(result: Any) -> _Usage:
    usage = getattr(result, "usage", None)
    if callable(usage):
        usage = usage()
    if usage is None or getattr(usage, "usage_available", None) is False:
        return _Usage()
    values = [getattr(usage, key, None) for key in ("input_tokens", "output_tokens", "total_tokens")]
    # Pydantic AI defaults missing provider counters to zero. Without an explicit
    # availability signal, treat an all-zero object as unknown, not free usage.
    if not any(values) and getattr(usage, "usage_available", None) is not True:
        return _Usage()
    input_tokens, output_tokens, total_tokens = (int(value) if value is not None else None for value in values)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return _Usage(input_tokens, output_tokens, total_tokens)


def _schema_version(output_type: Any) -> Optional[str]:
    if output_type is None:
        return None
    schema = getattr(output_type, "__name__", None)
    return str(schema) if schema else type(output_type).__name__


def _validation_result(error: Exception) -> str:
    return str(getattr(error, "_ainsight_validation_result", "request_failed"))


def _validation_result_from_result(result: Any) -> str:
    return str(getattr(result, "_ainsight_validation_result", "provider_validated"))


def _agent_name(task_type: str) -> str:
    root = task_type.split("_", 1)[0]
    return {"summary": "SummaryAgent", "graph": "GraphAgent", "qa": "QAAgent"}.get(root, "AinsightAgent")


def _value(instance: Any, name: str, default: Any) -> Any:
    return getattr(instance, name, default) if instance is not None else default
