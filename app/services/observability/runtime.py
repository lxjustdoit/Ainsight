"""Process lifecycle for logging, OpenTelemetry, and Langfuse reporting."""

import atexit
from copy import deepcopy
import logging
import sys
from threading import Lock
import traceback
from typing import Optional

from app.core.config import ObservabilityConfig
from app.services.observability.langfuse_client import (
    LangfuseEventSink,
    _configure_langfuse,
    _shutdown_langfuse,
)
from app.services.observability.logging import _configure_logging, get_logger, log_error
from app.services.observability.otel import _configure_otel, _shutdown_otel


# None: never started; True: starting/running under the lock; False: terminal.
running: Optional[bool] = None
_lifecycle_lock = Lock()
_active_config: Optional[ObservabilityConfig] = None
_active_sink: Optional[LangfuseEventSink] = None


def configure_observability(
    config: ObservabilityConfig,
) -> Optional[LangfuseEventSink]:
    """The sole application startup entry; failures propagate to the process entry point."""

    global running, _active_config, _active_sink
    with _lifecycle_lock:
        if running is False:
            raise RuntimeError("Observability has stopped or failed; restart the process")
        if running is True:
            if config != _active_config:
                raise RuntimeError("Observability is already running with different configuration")
            return _active_sink

        running = True
        try:
            # Snapshot nested mutable fields too, so later config edits are detected.
            _active_config = deepcopy(config)
            _configure_logging(_active_config.logging, service_name=_active_config.service_name)
            _configure_otel(_active_config.otel, service_name=_active_config.service_name)
            _active_sink = _configure_langfuse(
                _active_config.langfuse,
                service_name=_active_config.service_name,
            )
            atexit.register(shutdown_observability)
        except BaseException as exc:
            running = False
            _report_error("observability_initialization_failed", exc)
            # Do not re-enter the public shutdown method while holding this lock.
            _cleanup_resources()
            raise
        return _active_sink


def shutdown_observability() -> None:
    """Terminal, best-effort cleanup; the application owns process exit."""
    global running
    with _lifecycle_lock:
        if running is not True:
            return
        running = False
        _cleanup_resources()


def _cleanup_resources() -> None:
    global _active_sink
    # Drain Langfuse first so delivery-loss telemetry can still reach OTel.
    for name, close in (("langfuse", _shutdown_langfuse), ("otel", _shutdown_otel)):
        try:
            close()
        except BaseException as exc:
            _report_error("observability_shutdown_failed", exc, dependency=name)
    _active_sink = None


def _report_error(event: str, error: BaseException, **fields: str) -> None:
    """Reporting cleanup failures must never replace the original startup error."""
    try:
        logger = get_logger()
        # An unconfigured logger does not raise; avoid losing the message silently.
        if logger.hasHandlers() and logger.isEnabledFor(logging.ERROR):
            log_error(event, error, **fields)
            return
    except Exception:
        pass
    try:
        print(event, fields, file=sys.stderr)
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    except Exception:
        pass
