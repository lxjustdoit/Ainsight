"""Embedding provider requests and vector parsing; tracing uses observability."""

import json
from dataclasses import dataclass
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import EmbeddingModelConfig
from app.core.errors import ModelProviderError
from app.services.ai.utils.search_process_util import vector_cosine_similarity
from app.services.observability.llm_tracer import llm_tracer


Vector = list[float]


@dataclass(frozen=True)
class EmbeddingUsage:
    """Provider counters; absent usage must remain distinct from an explicit zero."""

    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def input_tokens(self) -> Optional[int]:
        return self.prompt_tokens

    @property
    def usage_available(self) -> bool:
        return self.prompt_tokens is not None or self.total_tokens is not None


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[Vector]
    usage: EmbeddingUsage = EmbeddingUsage()


class EmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError


class NoopEmbeddingClient(EmbeddingClient):
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=[])


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Send validated embedding batches and expose vectors plus provider usage."""

    def __init__(self, config: EmbeddingModelConfig):
        self.config = config

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        self._validate_config()
        normalized_inputs = [_trim_input(text, self.config.max_input_chars) for text in texts]
        if not normalized_inputs:
            return EmbeddingResult(vectors=[])

        return self._request_embeddings(texts=normalized_inputs, model_config=self.config)

    @llm_tracer("embedding", observation_type="embedding")
    def _request_embeddings(self, *, texts: Sequence[str], model_config: EmbeddingModelConfig) -> EmbeddingResult:
        """One real request, after validation and input trimming."""

        payload = {
            "model": model_config.model_name,
            "input": texts,
        }
        if model_config.dimensions is not None:
            payload["dimensions"] = model_config.dimensions

        request = Request(
            _embeddings_url(model_config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=model_config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ModelProviderError(f"Embedding request failed with status {exc.code}: {_read_error_body(exc)}") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ModelProviderError(f"Embedding request failed: {exc}") from exc

        return _parse_embedding_response(body, expected_count=len(texts))

    def _validate_config(self) -> None:
        if not self.config.enabled:
            raise ModelProviderError("Embedding model is disabled")
        if self.config.provider != "openai_compatible":
            raise ModelProviderError(f"Unsupported embedding provider: {self.config.provider}")
        if not self.config.model_name:
            raise ModelProviderError("Embedding model_name is required")
        if not self.config.api_key:
            raise ModelProviderError("Embedding api_key is required")
        if not self.config.base_url:
            raise ModelProviderError("Embedding base_url is required")


class RemoteEmbeddingSimilarityScorer:
    def __init__(self, client: EmbeddingClient):
        self.client = client

    def score(self, query: str, text: str) -> float | None:
        if not query.strip() or not text.strip():
            return None
        try:
            result = self.client.embed_texts([query, text])
        except ModelProviderError:
            return None
        if len(result.vectors) != 2:
            return None
        return vector_cosine_similarity(result.vectors[0], result.vectors[1])


def build_embedding_client(config: EmbeddingModelConfig) -> EmbeddingClient:
    if not config.enabled:
        return NoopEmbeddingClient()
    if config.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(config)
    raise ModelProviderError(f"Unsupported embedding provider: {config.provider}")


def _embeddings_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/embeddings"


def _trim_input(text: str, max_chars: int) -> str:
    return text.strip()[:max_chars]


def _parse_embedding_response(body: str, *, expected_count: int) -> EmbeddingResult:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelProviderError("Embedding response is not valid JSON") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelProviderError("Embedding response missing data list")

    indexed_items = []
    for fallback_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ModelProviderError("Embedding response data item must be an object")
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ModelProviderError("Embedding response data item missing embedding vector")
        indexed_items.append((int(item.get("index", fallback_index)), [_as_float(value) for value in vector]))

    vectors = [vector for _, vector in sorted(indexed_items, key=lambda item: item[0])]
    if len(vectors) != expected_count:
        raise ModelProviderError("Embedding response vector count does not match input count")

    usage = payload.get("usage") or {}
    return EmbeddingResult(
        vectors=vectors,
        usage=EmbeddingUsage(
            prompt_tokens=int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None,
            total_tokens=int(usage["total_tokens"]) if usage.get("total_tokens") is not None else None,
        ),
    )


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ModelProviderError("Embedding vector contains non-numeric value") from exc


def _read_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:500]
    except Exception:
        return ""
