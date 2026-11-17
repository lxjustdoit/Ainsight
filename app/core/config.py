from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml


BASE_CONFIG_PATHS = (
    "config.yaml",
    "config.yml",
    "config.json",
)

LOCAL_CONFIG_PATHS = (
    "config.local.yaml",
    "config.local.yml",
    "config.local.json",
)


@dataclass(frozen=True)
class CacheConfig:
    page_context_ttl_hours: int = 8
    summary_ttl_hours: int = 8
    graph_ttl_hours: int = 8
    qa_session_ttl_hours: int = 8


@dataclass(frozen=True)
class LoggingConfig:
    """JSON output settings; service identity belongs to ObservabilityConfig."""

    level: str = "INFO"
    max_field_length: int = 2000
    # Optional file path, including filename; None or empty uses stdout.
    output_path: Optional[str] = None


@dataclass(frozen=True)
class OTelTracesConfig:
    enabled: bool = True
    sampling_ratio: float = 1.0


@dataclass(frozen=True)
class OTelMetricsConfig:
    enabled: bool = False


@dataclass(frozen=True)
class OTelQueueConfig:
    max_queue_size: int = 2048
    schedule_delay_ms: int = 5000
    max_export_batch_size: int = 512
    export_timeout_ms: int = 30000


@dataclass(frozen=True)
class OTelConfig:
    enabled: bool = False
    exporter_endpoint: str = "http://localhost:4317"
    exporter_headers: dict[str, str] = field(default_factory=dict)
    exporter_insecure: bool = True
    exporter_protocol: str = "grpc"
    traces: OTelTracesConfig = field(default_factory=OTelTracesConfig)
    metrics: OTelMetricsConfig = field(default_factory=OTelMetricsConfig)
    queue: OTelQueueConfig = field(default_factory=OTelQueueConfig)


@dataclass(frozen=True)
class LangfuseQueueConfig:
    max_size: int = 10000
    enqueue_timeout_ms: int = 5
    flush_timeout_seconds: int = 5
    drop_when_full: bool = True


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""
    debug: bool = False
    flush_interval_ms: int = 10000
    queue: LangfuseQueueConfig = field(default_factory=LangfuseQueueConfig)


@dataclass(frozen=True)
class ObservabilityConfig:
    service_name: str = "ainsight-backend"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    otel: OTelConfig = field(default_factory=OTelConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)

    @property
    def langfuse_enabled(self) -> bool:
        return self.langfuse.enabled

    @property
    def signoz_enabled(self) -> bool:
        return self.otel.enabled


@dataclass(frozen=True)
class AIModelConfig:
    provider: str = "openai_compatible"
    model_name: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    prompt_version: str = "v1"
    structured_output: bool = False
    tools_enabled: bool = False
    target_output_tokens: int = 800
    max_output_tokens: int = 1200
    recent_qa_messages: int = 6
    temperature: float = 0.2
    input_token_price_per_1k: float = 0.0
    output_token_price_per_1k: float = 0.0
    search_findings_min_chars: int = 30
    search_findings_similarity_threshold: float = 0.12
    search_findings_max_results: int = 3
    search_findings_sentence_limit: int = 3
    search_findings_max_chars_per_result: int = 500
    search_findings_max_total_chars: int = 2500


@dataclass(frozen=True)
class EmbeddingModelConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    model_name: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout_seconds: float = 10.0
    dimensions: Optional[int] = None
    max_input_chars: int = 8000


@dataclass(frozen=True)
class AIConfig:
    summary: AIModelConfig
    graph: AIModelConfig
    qa: AIModelConfig
    embedding: EmbeddingModelConfig


@dataclass(frozen=True)
class AppConfig:
    cache: CacheConfig
    observability: ObservabilityConfig
    ai: AIConfig


def _env(name: str, default: Any) -> Any:
    value = os.getenv(name)
    return default if value is None else value


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _float_env_with_alias(name: str, alias: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(alias)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _optional_int_env(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _dict_env(name: str, default: dict[str, str]) -> dict[str, str]:
    value = os.getenv(name)
    if value is None:
        return dict(default)
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


def _read_single_config_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(config_file) or {}
        if path.endswith(".json"):
            return json.load(config_file)
    raise ValueError("Config file must be .yaml, .yml, or .json")


def _merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _first_existing(paths: tuple[str, ...]) -> Optional[str]:
    for candidate in paths:
        if Path(candidate).is_file():
            return candidate
    return None


def _read_config_file(path: Optional[str]) -> dict:
    explicit = path or os.getenv("AINSIGHT_CONFIG_FILE")
    if explicit:
        return _read_single_config_file(explicit)

    base_path = _first_existing(BASE_CONFIG_PATHS)
    local_path = _first_existing(LOCAL_CONFIG_PATHS)
    raw = _read_single_config_file(base_path) if base_path else {}
    if local_path:
        raw = _merge_config(raw, _read_single_config_file(local_path))
    return raw


def _model_config(raw: dict, task_name: str) -> AIModelConfig:
    task = raw.get("ai", {}).get(task_name, {})
    upper = task_name.upper()
    return AIModelConfig(
        provider=os.getenv(f"{upper}_LLM_PROVIDER", task.get("provider", "openai_compatible")),
        model_name=os.getenv(f"{upper}_LLM_MODEL", task.get("model_name", "")),
        base_url=os.getenv(f"{upper}_LLM_BASE_URL", task.get("base_url", "https://api.openai.com/v1")),
        api_key=os.getenv(f"{upper}_LLM_API_KEY", task.get("api_key", "")),
        prompt_version=os.getenv(f"{upper}_PROMPT_VERSION", task.get("prompt_version", f"{task_name}_v1")),
        structured_output=_bool_env(f"{upper}_STRUCTURED_OUTPUT", bool(task.get("structured_output", False))),
        tools_enabled=_bool_env(f"{upper}_TOOLS_ENABLED", bool(task.get("tools_enabled", False))),
        target_output_tokens=_int_env(
            f"{upper}_TARGET_OUTPUT_TOKENS", int(task.get("target_output_tokens", 800))
        ),
        max_output_tokens=_int_env(f"{upper}_MAX_OUTPUT_TOKENS", int(task.get("max_output_tokens", 1200))),
        recent_qa_messages=_int_env(f"{upper}_RECENT_QA_MESSAGES", int(task.get("recent_qa_messages", 6))),
        temperature=_float_env(f"{upper}_TEMPERATURE", float(task.get("temperature", 0.2))),
        input_token_price_per_1k=_float_env(
            f"{upper}_INPUT_TOKEN_PRICE_PER_1K", float(task.get("input_token_price_per_1k", 0.0))
        ),
        output_token_price_per_1k=_float_env(
            f"{upper}_OUTPUT_TOKEN_PRICE_PER_1K", float(task.get("output_token_price_per_1k", 0.0))
        ),
        search_findings_min_chars=_int_env(
            f"{upper}_SEARCH_FINDINGS_MIN_CHARS", int(task.get("search_findings_min_chars", 30))
        ),
        search_findings_similarity_threshold=_float_env_with_alias(
            f"{upper}_SEARCH_FINDINGS_SIMILARITY_THRESHOLD",
            f"{upper}_SEARCH_FINDINGS_EMBEDDING_THRESHOLD",
            float(
                task.get(
                    "search_findings_similarity_threshold",
                    task.get("search_findings_embedding_threshold", 0.12),
                )
            ),
        ),
        search_findings_max_results=_int_env(
            f"{upper}_SEARCH_FINDINGS_MAX_RESULTS", int(task.get("search_findings_max_results", 3))
        ),
        search_findings_sentence_limit=_int_env(
            f"{upper}_SEARCH_FINDINGS_SENTENCE_LIMIT", int(task.get("search_findings_sentence_limit", 3))
        ),
        search_findings_max_chars_per_result=_int_env(
            f"{upper}_SEARCH_FINDINGS_MAX_CHARS_PER_RESULT",
            int(task.get("search_findings_max_chars_per_result", 500)),
        ),
        search_findings_max_total_chars=_int_env(
            f"{upper}_SEARCH_FINDINGS_MAX_TOTAL_CHARS", int(task.get("search_findings_max_total_chars", 2500))
        ),
    )


def _embedding_config(raw: dict) -> EmbeddingModelConfig:
    config = raw.get("ai", {}).get("embedding", {})
    return EmbeddingModelConfig(
        enabled=_bool_env("EMBEDDING_ENABLED", bool(config.get("enabled", False))),
        provider=os.getenv("EMBEDDING_PROVIDER", config.get("provider", "openai_compatible")),
        model_name=os.getenv("EMBEDDING_MODEL", config.get("model_name", "")),
        base_url=os.getenv("EMBEDDING_BASE_URL", config.get("base_url", "https://api.openai.com/v1")),
        api_key=os.getenv("EMBEDDING_API_KEY", config.get("api_key", "")),
        timeout_seconds=_float_env("EMBEDDING_TIMEOUT_SECONDS", float(config.get("timeout_seconds", 10.0))),
        dimensions=_optional_int_env("EMBEDDING_DIMENSIONS", config.get("dimensions")),
        max_input_chars=_int_env("EMBEDDING_MAX_INPUT_CHARS", int(config.get("max_input_chars", 8000))),
    )


def _observability_config(raw: dict) -> ObservabilityConfig:
    config = raw.get("observability", {})
    return ObservabilityConfig(
        service_name=_env("SERVICE_NAME", config.get("service_name", "ainsight-backend")),
        logging=_logging_config(config),
        otel=_otel_config(config),
        langfuse=_langfuse_config(config),
    )


def _logging_config(config: dict) -> LoggingConfig:
    logging_config = config.get("logging", {})
    return LoggingConfig(
        level=str(_env("LOG_LEVEL", logging_config.get("level", config.get("log_level", "INFO")))),
        max_field_length=_int_env("LOG_MAX_FIELD_LENGTH", int(logging_config.get("max_field_length", 2000))),
        output_path=_env("LOG_OUTPUT_PATH", logging_config.get("output_path")) or None,
    )


def _otel_config(config: dict) -> OTelConfig:
    otel = config.get("otel", {})
    traces = otel.get("traces", {})
    metrics = otel.get("metrics", {})
    queue = otel.get("queue", {})
    return OTelConfig(
        enabled=_bool_env("OTEL_ENABLED", bool(otel.get("enabled", config.get("signoz_enabled", False)))),
        exporter_endpoint=_env("OTEL_EXPORTER_ENDPOINT", otel.get("exporter_endpoint", "http://localhost:4317")),
        exporter_headers=_dict_env("OTEL_EXPORTER_HEADERS", otel.get("exporter_headers", {}) or {}),
        exporter_insecure=_bool_env("OTEL_EXPORTER_INSECURE", bool(otel.get("exporter_insecure", True))),
        exporter_protocol=str(_env("OTEL_EXPORTER_PROTOCOL", otel.get("exporter_protocol", "grpc"))),
        traces=OTelTracesConfig(
            enabled=_bool_env("OTEL_TRACES_ENABLED", bool(traces.get("enabled", True))),
            sampling_ratio=_float_env("OTEL_TRACES_SAMPLING_RATIO", float(traces.get("sampling_ratio", 1.0))),
        ),
        metrics=OTelMetricsConfig(
            enabled=_bool_env("OTEL_METRICS_ENABLED", bool(metrics.get("enabled", False))),
        ),
        queue=OTelQueueConfig(
            max_queue_size=_int_env("OTEL_QUEUE_MAX_SIZE", int(queue.get("max_queue_size", 2048))),
            schedule_delay_ms=_int_env("OTEL_QUEUE_SCHEDULE_DELAY_MS", int(queue.get("schedule_delay_ms", 5000))),
            max_export_batch_size=_int_env(
                "OTEL_QUEUE_MAX_EXPORT_BATCH_SIZE", int(queue.get("max_export_batch_size", 512))
            ),
            export_timeout_ms=_int_env("OTEL_QUEUE_EXPORT_TIMEOUT_MS", int(queue.get("export_timeout_ms", 30000))),
        ),
    )


def _langfuse_config(config: dict) -> LangfuseConfig:
    langfuse = config.get("langfuse", {})
    queue = langfuse.get("queue", {})
    return LangfuseConfig(
        enabled=_bool_env("LANGFUSE_ENABLED", bool(langfuse.get("enabled", config.get("langfuse_enabled", False)))),
        host=str(_env("LANGFUSE_HOST", langfuse.get("host", "http://localhost:3000"))),
        public_key=str(_env("LANGFUSE_PUBLIC_KEY", langfuse.get("public_key", ""))),
        secret_key=str(_env("LANGFUSE_SECRET_KEY", langfuse.get("secret_key", ""))),
        debug=_bool_env("LANGFUSE_DEBUG", bool(langfuse.get("debug", False))),
        flush_interval_ms=_int_env(
            "LANGFUSE_FLUSH_INTERVAL_MS",
            int(langfuse.get("flush_interval_ms", langfuse.get("flush_interval", 10000))),
        ),
        queue=LangfuseQueueConfig(
            max_size=_int_env("LANGFUSE_QUEUE_MAX_SIZE", int(queue.get("max_size", 10000))),
            enqueue_timeout_ms=_int_env(
                "LANGFUSE_QUEUE_ENQUEUE_TIMEOUT_MS", int(queue.get("enqueue_timeout_ms", 5))
            ),
            flush_timeout_seconds=_int_env(
                "LANGFUSE_QUEUE_FLUSH_TIMEOUT_SECONDS", int(queue.get("flush_timeout_seconds", 5))
            ),
            drop_when_full=_bool_env("LANGFUSE_QUEUE_DROP_WHEN_FULL", bool(queue.get("drop_when_full", True))),
        ),
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    raw = _read_config_file(config_path)
    cache = raw.get("cache", {})
    return AppConfig(
        cache=CacheConfig(
            page_context_ttl_hours=_int_env(
                "PAGE_CONTEXT_TTL_HOURS", int(cache.get("page_context_ttl_hours", 8))
            ),
            summary_ttl_hours=_int_env("SUMMARY_TTL_HOURS", int(cache.get("summary_ttl_hours", 8))),
            graph_ttl_hours=_int_env("GRAPH_TTL_HOURS", int(cache.get("graph_ttl_hours", 8))),
            qa_session_ttl_hours=_int_env("QA_SESSION_TTL_HOURS", int(cache.get("qa_session_ttl_hours", 8))),
        ),
        observability=_observability_config(raw),
        ai=AIConfig(
            summary=_model_config(raw, "summary"),
            graph=_model_config(raw, "graph"),
            qa=_model_config(raw, "qa"),
            embedding=_embedding_config(raw),
        ),
    )
