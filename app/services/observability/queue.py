"""Bounded observability event queues with replaceable delivery backends."""

import logging
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Callable, Generic, Optional, Protocol, TypeVar


EventT = TypeVar("EventT")


class ObservabilityQueue(Protocol[EventT]):
    def enqueue(self, event: EventT) -> bool:
        ...

    def flush(self, timeout_seconds: float) -> bool:
        ...

    def shutdown(self, timeout_seconds: float) -> bool:
        ...


class DirectObservabilityQueue(Generic[EventT]):
    """Synchronous queue used by no-op mode and deterministic tests."""

    def __init__(self, consumer: Callable[[EventT], None]):
        self._consumer = consumer

    def enqueue(self, event: EventT) -> bool:
        self._consumer(event)
        return True

    def flush(self, timeout_seconds: float) -> bool:
        return True

    def shutdown(self, timeout_seconds: float) -> bool:
        return True


class InMemoryObservabilityQueue(Generic[EventT]):
    """Best-effort bounded queue that keeps exporter work off request threads."""

    def __init__(
        self,
        consumer: Callable[[EventT], None],
        *,
        max_size: int,
        enqueue_timeout_ms: int,
        drop_when_full: bool,
        name: str,
        on_drop: Optional[Callable[[EventT, str], None]] = None,
    ):
        self._consumer = consumer
        self._on_drop = on_drop
        self._queue: Queue[Optional[EventT]] = Queue(maxsize=max(1, max_size))
        self._enqueue_timeout = max(0, enqueue_timeout_ms) / 1000.0
        self._drop_when_full = drop_when_full
        self._stopped = Event()
        self._state_lock = Lock()
        self._logger = logging.getLogger("ainsight.observability.queue")
        self._worker = Thread(target=self._run, name=name, daemon=True)
        self._worker.start()

    def enqueue(self, event: EventT) -> bool:
        # Shutdown must not race with an enqueue after the worker has exited.
        with self._state_lock:
            if self._stopped.is_set():
                reason = "queue_stopped"
            else:
                try:
                    if self._drop_when_full:
                        self._queue.put_nowait(event)
                    else:
                        self._queue.put(event, timeout=self._enqueue_timeout)
                    return True
                except Full:
                    reason = "queue_full"
        self._drop(event, reason)
        return False

    def _drop(self, event: EventT, reason: str) -> None:
        """Report terminal loss once; reporting must not fail the business request."""
        from app.services.observability.context import observability_scope

        with observability_scope(trace_id=getattr(event, "trace_id", None), llm_call_id=getattr(event, "llm_call_id", None)):
            self._logger.warning("observability_event_dropped", extra={"structured_fields": {"reason": reason}})
            if self._on_drop is not None:
                try:
                    self._on_drop(event, reason)
                except Exception:
                    self._logger.exception("observability_drop_reporting_failed")

    def flush(self, timeout_seconds: float) -> bool:
        deadline = monotonic() + max(0.0, timeout_seconds)
        while self._queue.unfinished_tasks and monotonic() < deadline:
            sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def shutdown(self, timeout_seconds: float) -> bool:
        with self._state_lock:
            if self._stopped.is_set():
                return not self._worker.is_alive()
            self._stopped.set()
        started_at = monotonic()
        drained = self.flush(timeout_seconds)
        if drained:
            try:
                self._queue.put_nowait(None)
            except Full:
                drained = False
        remaining = max(0.0, timeout_seconds - (monotonic() - started_at))
        self._worker.join(timeout=remaining)
        return drained and not self._worker.is_alive()

    def _run(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=0.25)
            except Empty:
                if self._stopped.is_set():
                    return
                continue
            try:
                if event is None:
                    return
                self._consumer(event)
            except Exception as exc:
                self._drop(event, "export_failed")
                from app.services.observability.context import observability_scope
                from app.services.observability.logging import log_error

                with observability_scope(
                    trace_id=getattr(event, "trace_id", None),
                    llm_call_id=getattr(event, "llm_call_id", None),
                ):
                    log_error(
                        "dependency_error",
                        exc,
                        dependency="observability_exporter",
                        error_location=f"ObservabilityQueue.{self._worker.name}",
                    )
            finally:
                self._queue.task_done()
