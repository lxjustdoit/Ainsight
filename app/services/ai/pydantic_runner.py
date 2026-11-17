from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Optional, Type

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.errors import ModelProviderError
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.output_repair import OutputRepairFailure, append_regeneration_feedback, parse_repaired_output
from app.services.observability.llm_tracer import llm_tracer
from app.services.observability.context import trace_scope_if_missing


@dataclass(frozen=True)
class PydanticAIUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class PydanticAIResult:
    output: Any
    usage: PydanticAIUsage


@dataclass(frozen=True)
class PydanticAITextStreamEvent:
    event: str
    content: str = ""
    usage: PydanticAIUsage = field(default_factory=PydanticAIUsage)


@dataclass(frozen=True)
class _ValidatedProviderResult:
    """Keep provider output for tracing while returning locally validated output."""

    output: Any
    usage: Any
    _ainsight_raw_output: Any
    _ainsight_validation_result: str = "passed"


class PydanticAIRunner:
    def run(
        self,
        *,
        model_config: AgentModelConfig,
        output_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> PydanticAIResult:
        raise NotImplementedError

    def run_text(
        self,
        *,
        model_config: AgentModelConfig,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> PydanticAIResult:
        raise NotImplementedError

    def run_text_stream(
        self,
        *,
        model_config: AgentModelConfig,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> Iterable[PydanticAITextStreamEvent]:
        raise NotImplementedError


class PydanticAIAgentRunner(PydanticAIRunner):
    def __init__(self, model: Optional[Any] = None):
        self.model = model

    def run(
        self,
        *,
        model_config: AgentModelConfig,
        output_type: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> PydanticAIResult:
        with trace_scope_if_missing():
            model = self.model or self._build_model(model_config)
            request_output_type = self._request_output_type(model_config, output_type)
            try:
                output, usage = self._request_llm_once(
                    model=model,
                    model_config=model_config,
                    output_type=request_output_type,
                    schema_output_type=output_type,
                    validation_output_type=None if model_config.structured_output else output_type,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    capabilities=capabilities,
                    attempt=retry_count,
                )
            except OutputRepairFailure as exc:
                usage = self._usage_from_failed_validation(exc, model_config)
                regenerated_prompt = append_regeneration_feedback(user_prompt, exc.feedback)
            else:
                return PydanticAIResult(output=output, usage=usage)

            retry_output, retry_usage = self._request_llm_once(
                model=model,
                model_config=model_config,
                output_type=request_output_type,
                schema_output_type=output_type,
                validation_output_type=output_type,
                system_prompt=system_prompt,
                user_prompt=regenerated_prompt,
                capabilities=capabilities,
                attempt=retry_count + 1,
            )
            return PydanticAIResult(output=retry_output, usage=_add_usage(usage, retry_usage))

    def run_text(
        self,
        *,
        model_config: AgentModelConfig,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> PydanticAIResult:
        with trace_scope_if_missing():
            model = self.model or self._build_model(model_config)
            output, usage = self._request_llm_once(
                model=model,
                model_config=model_config,
                output_type=str,
                schema_output_type=None,
                validation_output_type=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                capabilities=capabilities,
                attempt=retry_count,
            )
            return PydanticAIResult(output=str(output).strip(), usage=usage)

    def run_text_stream(
        self,
        *,
        model_config: AgentModelConfig,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        retry_count: int = 0,
    ) -> Iterable[PydanticAITextStreamEvent]:
        with trace_scope_if_missing():
            model = self.model or self._build_model(model_config)
            yield from self._start_stream_call(
                model=model,
                model_config=model_config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                capabilities=capabilities,
                attempt=retry_count,
            )

    @llm_tracer()
    def _start_stream_call(
        self,
        *,
        model: Any,
        model_config: AgentModelConfig,
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        attempt: int = 0,
    ) -> Iterable[PydanticAITextStreamEvent]:
        agent = Agent(
            model,
            output_type=str,
            system_prompt=system_prompt,
            retries=0,
            capabilities=capabilities or [],
        )
        stream = agent.run_stream_sync(
            user_prompt,
            model_settings={
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_output_tokens,
            },
        )
        chunks: list[str] = []
        for chunk in stream.stream_text(delta=True, debounce_by=None):
            chunks.append(chunk)
            yield PydanticAITextStreamEvent(event="delta", content=chunk)
        yield PydanticAITextStreamEvent(
            event="final",
            content="".join(chunks).strip(),
            usage=self._usage_from_result(stream.usage, model_config),
        )

    def _request_output_type(self, model_config: AgentModelConfig, output_type: Type[BaseModel]) -> Any:
        if model_config.structured_output:
            return NativeOutput(output_type, strict=True)
        return str

    def _request_llm_once(
        self,
        *,
        model: Any,
        model_config: AgentModelConfig,
        output_type: Any,
        schema_output_type: Optional[Type[BaseModel]],
        validation_output_type: Optional[Type[BaseModel]],
        system_prompt: str,
        user_prompt: str,
        capabilities: Optional[list[Any]] = None,
        attempt: int = 0,
    ) -> tuple[Any, PydanticAIUsage]:
        agent = Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=0,
            capabilities=capabilities or [],
        )
        result = self._run_with_request_retry(
            agent,
            user_prompt,
            model_config,
            system_prompt=system_prompt,
            attempt=attempt,
            schema_output_type=schema_output_type,
            validation_output_type=validation_output_type,
        )
        usage = self._usage_from_result(result.usage, model_config)
        return result.output, usage

    def _run_with_request_retry(
        self,
        agent: Agent,
        user_prompt: str,
        model_config: AgentModelConfig,
        *,
        system_prompt: str,
        attempt: int,
        schema_output_type: Optional[Type[BaseModel]],
        validation_output_type: Optional[Type[BaseModel]],
    ) -> Any:
        try:
            return self._start_llm_call(
                agent=agent,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model_config=model_config,
                attempt=attempt,
                schema_output_type=schema_output_type,
                validation_output_type=validation_output_type,
            )
        except Exception as exc:
            if not _is_retryable_request_error(exc):
                raise
            return self._start_llm_call(
                agent=agent,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model_config=model_config,
                attempt=attempt + 1,
                schema_output_type=schema_output_type,
                validation_output_type=validation_output_type,
            )

    @llm_tracer()
    def _start_llm_call(
        self,
        *,
        agent: Agent,
        user_prompt: str,
        system_prompt: str,
        model_config: AgentModelConfig,
        attempt: int = 0,
        schema_output_type: Optional[Type[BaseModel]] = None,
        validation_output_type: Optional[Type[BaseModel]] = None,
    ) -> Any:
        result = agent.run_sync(
            user_prompt,
            model_settings={
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_output_tokens,
            },
        )
        if validation_output_type is None:
            return result
        try:
            parsed_output = parse_repaired_output(
                result.output,
                output_type=validation_output_type,
                task_type=model_config.task_type,
            )
        except OutputRepairFailure as exc:
            exc._ainsight_provider_result = result
            exc._ainsight_raw_response = str(result.output)
            exc._ainsight_validation_result = "failed"
            raise
        return _ValidatedProviderResult(
            output=parsed_output,
            usage=result.usage,
            _ainsight_raw_output=result.output,
        )

    def _usage_from_failed_validation(
        self,
        error: OutputRepairFailure,
        model_config: AgentModelConfig,
    ) -> PydanticAIUsage:
        provider_result = getattr(error, "_ainsight_provider_result", None)
        if provider_result is None:
            return PydanticAIUsage()
        return self._usage_from_result(provider_result.usage, model_config)

    def _usage_from_result(self, usage: Any, model_config: AgentModelConfig) -> PydanticAIUsage:
        usage = usage() if callable(usage) else usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        # Cost aggregation is handled by Langfuse. Keep the field at 0.0 for API compatibility.
        # cost = (input_tokens / 1000.0 * model_config.input_token_price_per_1k) + (
        #     output_tokens / 1000.0 * model_config.output_token_price_per_1k
        # )
        return PydanticAIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(getattr(usage, "total_tokens", None) or input_tokens + output_tokens),
            cost=0.0,
        )

    def _build_model(self, model_config: AgentModelConfig) -> Any:
        if model_config.provider in {"openai", "openai_compatible"}:
            return self._build_openai_model(model_config)
        if model_config.provider in {"anthropic", "anthropic_compatible", "minimax_anthropic"}:
            return self._build_anthropic_model(model_config)
        raise ModelProviderError(f"Unsupported Pydantic AI provider: {model_config.provider}")

    def _build_openai_model(self, model_config: AgentModelConfig) -> OpenAIChatModel:
        if not model_config.model_name:
            raise ModelProviderError("LLM model_name is required")
        if not model_config.api_key:
            raise ModelProviderError("LLM api_key is required")
        provider = OpenAIProvider(base_url=model_config.base_url, api_key=model_config.api_key)
        return OpenAIChatModel(model_config.model_name, provider=provider)

    def _build_anthropic_model(self, model_config: AgentModelConfig) -> AnthropicModel:
        if not model_config.model_name:
            raise ModelProviderError("LLM model_name is required")
        if not model_config.api_key:
            raise ModelProviderError("LLM api_key is required")
        if not model_config.base_url:
            raise ModelProviderError("LLM base_url is required")
        provider = AnthropicProvider(base_url=model_config.base_url, api_key=model_config.api_key)
        return AnthropicModel(model_config.model_name, provider=provider)


def _add_usage(first: PydanticAIUsage, second: PydanticAIUsage) -> PydanticAIUsage:
    return PydanticAIUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        cost=0.0,
    )


def _is_retryable_request_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = _extract_status_code(str(exc))
    if status_code in {401, 402, 403, 400}:
        return False
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return True
    error_text = str(exc).lower()
    retryable_markers = ("timeout", "temporarily", "connection", "rate limit", "server error")
    non_retryable_markers = ("insufficient balance", "unauthorized", "forbidden", "invalid request")
    if any(marker in error_text for marker in non_retryable_markers):
        return False
    return any(marker in error_text for marker in retryable_markers)


def _extract_status_code(text: str) -> Optional[int]:
    match = re.search(r"status_code:\s*(\d{3})", text)
    if not match:
        return None
    return int(match.group(1))
