import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
from urllib import error, request

from app.core.errors import AIOutputValidationError, ModelProviderError
from app.services.ai.model_config import ModelConfig


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    duration_ms: int = 0


class LLMClient(Protocol):
    def complete(self, *, model_config: ModelConfig, messages: List[LLMMessage]) -> LLMResponse:
        ...


class OpenAICompatibleLLMClient:
    """Minimal OpenAI-compatible chat completions client using stdlib HTTP."""

    def complete(self, *, model_config: ModelConfig, messages: List[LLMMessage]) -> LLMResponse:
        if not model_config.model_name:
            raise ModelProviderError("LLM model_name is required")
        if not model_config.api_key:
            raise ModelProviderError("LLM api_key is required")
        url = model_config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_config.model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
            },
        )
        started_at = time.perf_counter()
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelProviderError(f"LLM provider HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise ModelProviderError(f"LLM provider request failed: {exc.reason}") from exc

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelProviderError("LLM provider returned an unsupported response shape") from exc
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        #需要核实一下官方文档，是否需要自己计算费用
        cost = (input_tokens / 1000.0 * model_config.input_token_price_per_1k) + (
            output_tokens / 1000.0 * model_config.output_token_price_per_1k
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("total_tokens", 0) or input_tokens + output_tokens),
            cost=cost,
            duration_ms=duration_ms,
        )


class StaticLLMClient:
    """Test helper that returns queued JSON responses and records prompts."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, *, model_config: ModelConfig, messages: List[LLMMessage]) -> LLMResponse:
        self.calls.append({"model_config": model_config, "messages": messages})
        if not self.responses:
            raise ModelProviderError("StaticLLMClient has no response queued")
        content = json.dumps(self.responses.pop(0), ensure_ascii=False)
        return LLMResponse(content=content, input_tokens=10, output_tokens=20, total_tokens=30, duration_ms=1)


def parse_json_object(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIOutputValidationError("LLM output is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AIOutputValidationError("LLM output must be a JSON object")
    return value


def require_string(data: Dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AIOutputValidationError(f"LLM output field '{field}' must be a non-empty string")
    return value.strip()


def require_list(data: Dict[str, Any], field: str) -> List[Any]:
    value = data.get(field)
    if not isinstance(value, list):
        raise AIOutputValidationError(f"LLM output field '{field}' must be a list")
    return value
