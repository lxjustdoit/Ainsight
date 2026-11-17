"""Python logging setup with JSON output and trace correlation."""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from app.core.config import LoggingConfig, ObservabilityConfig
from app.services.observability.context import correlation_fields


LOGGER_NAME = "ainsight"
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "cookie",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "secret_key",
    "set_cookie",
    "token",
    "x_api_key",
}
_CONTENT_KEYS = {
    "page_content",
    "pagecontent",
    "selected_text",
    "selectedtext",
    "question",
    "prompt",
    "response",
    "user_prompt",
    "system_prompt",
}
_SENSITIVE_COMPACT_KEYS = {key.replace("_", "") for key in _SENSITIVE_KEYS}
_CONTENT_COMPACT_KEYS = {key.replace("_", "") for key in _CONTENT_KEYS}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[^\s,;]+"),
    re.compile(
        r"(?i)\b(authorization|x-api-key|api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"id[_-]?token|client[_-]?secret|secret[_-]?key|password|set-cookie|cookie)"
        r"\s*([:=])\s*([^\s,;]+)"
    ),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
)
_RESERVED_FIELDS = {"ts", "level", "event", "status", "trace_id", "span_id", "service_name"}
_CORRELATION_IDS = {
    "trace_id", "span_id", "user_id", "ctx", "summary_id", "graph_id",
    "conversation_id", "request_id", "external_request_id", "llm_call_id",
}


class JsonLogFormatter(logging.Formatter):
    """Serialize standard LogRecord instances into the service JSON schema."""

    def __init__(
        self,
        max_field_length: int = 2000,
        *,
        service_name: str = ObservabilityConfig.service_name,
    ):
        super().__init__()
        self.max_field_length = max(1, max_field_length)
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", None) or record.getMessage()
        fields = {
            key: value
            for key, value in (getattr(record, "structured_fields", {}) or {}).items()
            if key not in _RESERVED_FIELDS
        }
        payload: dict[str, Any] = {
            **fields,
            **correlation_fields(),
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "event": event,
        }
        status = getattr(record, "status", None)
        if status:
            payload["status"] = status
        identifiers = {key: payload[key] for key in _CORRELATION_IDS if key in payload}
        payload = _sanitize(payload, self.max_field_length)
        # Truncating identifiers breaks Collector parsing and cross-system links.
        payload.update(identifiers)
        # Keep the shared service identity and full exception chain intact.
        payload["service_name"] = self.service_name
        if record.exc_info:
            payload["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _configure_logging(
    config: LoggingConfig,
    *,
    service_name: str = ObservabilityConfig.service_name,
    stream: Optional[TextIO] = None,
) -> logging.Logger:
    """Internal startup step; a configured file takes precedence over stream."""

    level = _log_level(config.level)
    formatter = JsonLogFormatter(config.max_field_length, service_name=service_name)
    # Open the replacement before touching the current logger so failures leave it usable.
    handler: logging.Handler
    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output_path, mode="a", encoding="utf-8")
    else:
        handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    try:
        handler._ainsight_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(formatter)
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(level)
        logger.propagate = False
        for previous_handler in list(logger.handlers):
            if getattr(previous_handler, "_ainsight_handler", False):
                logger.removeHandler(previous_handler)
                previous_handler.close()
        logger.addHandler(handler)
    except BaseException:
        handler.close()
        raise
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def log_info(event: str, status: str = "success", **fields: Any) -> None:
    get_logger().info(
        event,
        extra={"event": event, "status": status, "structured_fields": fields},
    )


def log_error(event: str, error: BaseException, status: str = "failed", **fields: Any) -> None:
    error_fields = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **fields,
    }
    get_logger().error(
        event,
        extra={"event": event, "status": status, "structured_fields": error_fields},
        exc_info=(type(error), error, error.__traceback__),
    )


def redact_sensitive_text(value: Any, *, max_length: Optional[int] = None) -> str:
    """Compatibility hook for telemetry redaction, paused during local debugging."""

    # Privacy redaction is temporarily paused for development debugging.
    # Restore with: telemetry_text = _redact_string(str(value))
    telemetry_text = str(value)
    if max_length is not None and len(telemetry_text) > max(1, max_length):
        return f"{telemetry_text[:max(1, max_length)]}...[truncated]"
    return telemetry_text


def sanitize_telemetry_fields(fields: dict[str, Any], *, max_length: int = 2000) -> dict[str, Any]:
    """Normalize custom telemetry fields; privacy redaction is currently paused."""

    return _sanitize(fields, max(1, max_length))


def _sanitize(value: Any, max_length: int, key: str = "") -> Any:
    # Privacy redaction is temporarily paused for development debugging.
    # Restore the following block when privacy handling resumes:
    # normalized_key = key.lower().replace("-", "_")
    # compact_key = normalized_key.replace("_", "")
    # if _is_sensitive_key(normalized_key, compact_key):
    #     return "[REDACTED]"
    # if _is_content_key(normalized_key, compact_key):
    #     return "[NOT_LOGGED]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize(item, max_length, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item, max_length, key) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, max_length=max_length)
    return value


def _is_sensitive_key(normalized_key: str, compact_key: str) -> bool:
    if normalized_key in _SENSITIVE_KEYS or compact_key in _SENSITIVE_COMPACT_KEYS:
        return True
    return normalized_key.endswith(("_api_key", "_access_token", "_refresh_token", "_secret", "_password"))


def _is_content_key(normalized_key: str, compact_key: str) -> bool:
    if normalized_key in _CONTENT_KEYS or compact_key in _CONTENT_COMPACT_KEYS:
        return True
    return normalized_key.endswith(("_prompt", "_prompt_text", "_response_text", "_page_content"))


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.groups == 1:
            redacted = pattern.sub(r"\1 [REDACTED]", redacted)
        elif pattern.groups == 3:
            redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _log_level(value: str) -> int:
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {value}")
    return level
