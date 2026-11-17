"""Offline checks for the process-terminal observability lifecycle and entry points."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
import importlib
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock, current_thread
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, call, patch

from app.core.config import (
    AIConfig, AIModelConfig, AppConfig, CacheConfig, EmbeddingModelConfig,
    LangfuseConfig, LangfuseQueueConfig, ObservabilityConfig, OTelConfig,
)
from app.services.observability import langfuse_client, runtime
from app.services.observability.queue import InMemoryObservabilityQueue


COMPONENTS = ("logging", "otel", "langfuse")
CLEANUP_ORDER = ("langfuse", "otel")
WAIT_SECONDS = 5
BACKEND_ROOT = Path(__file__).resolve().parents[1]


class _ObservedLock:
    """Expose actual lock contention without timing sleeps in concurrency checks."""

    def __init__(self):
        self.lock = Lock()
        self.contended = Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


@dataclass
class _RuntimeMocks:
    configure: dict[str, MagicMock]
    shutdown: dict[str, MagicMock]
    report: MagicMock
    logger: MagicMock
    register: MagicMock
    calls: MagicMock


@contextmanager
def _isolated_runtime():
    # Restore the four lifecycle fields without registering real exit callbacks.
    with ExitStack() as stack:
        stack.enter_context(patch.multiple(
            runtime, running=None, _lifecycle_lock=Lock(), _active_config=None,
            _active_sink=None,
        ))
        calls = MagicMock()
        configure = {}
        shutdown = {}
        for name in COMPONENTS:
            configure[name] = stack.enter_context(patch.object(runtime, f"_configure_{name}"))
            calls.attach_mock(configure[name], f"configure_{name}")
        for name in CLEANUP_ORDER:
            shutdown[name] = stack.enter_context(patch.object(runtime, f"_shutdown_{name}"))
            calls.attach_mock(shutdown[name], f"shutdown_{name}")
        logger = stack.enter_context(patch.object(runtime, "get_logger")).return_value
        logger.hasHandlers.return_value = True
        logger.isEnabledFor.return_value = True
        report = stack.enter_context(patch.object(runtime, "log_error"))
        calls.attach_mock(report, "report")
        register = stack.enter_context(patch.object(runtime.atexit, "register"))
        stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        yield _RuntimeMocks(configure, shutdown, report, logger, register, calls)


def _app_config():
    model = AIModelConfig(model_name="offline-lifecycle-test")
    return AppConfig(
        cache=CacheConfig(), observability=ObservabilityConfig(),
        ai=AIConfig(summary=model, graph=model, qa=model, embedding=EmbeddingModelConfig()),
    )


class RuntimeLifecycleTests(unittest.TestCase):
    def assert_no_cleanup(self, mocks):
        for shutdown in mocks.shutdown.values():
            shutdown.assert_not_called()

    def assert_cleanup_once(self, mocks):
        for shutdown in mocks.shutdown.values():
            shutdown.assert_called_once_with()
        names = [entry[0] for entry in mocks.calls.mock_calls if entry[0].startswith("shutdown_")]
        self.assertEqual(names, [f"shutdown_{name}" for name in CLEANUP_ORDER])
        self.assertIsNone(runtime._active_sink)

    def test_equal_config_is_idempotent_and_registers_shutdown_once(self):
        with _isolated_runtime() as mocks:
            config = ObservabilityConfig()
            expected = mocks.configure["langfuse"].return_value
            for _ in range(2):
                self.assertIs(runtime.configure_observability(replace(config)), expected)
            for name in COMPONENTS:
                mocks.configure[name].assert_called_once_with(
                    getattr(config, name), service_name=config.service_name,
                )
            mocks.register.assert_called_once_with(runtime.shutdown_observability)
            self.assertIs(runtime.running, True)
            self.assert_no_cleanup(mocks)

    def test_running_is_true_before_each_initializer_under_the_lifecycle_lock(self):
        with _isolated_runtime() as mocks:
            def initialize(*args, **kwargs):
                self.assertIs(runtime.running, True)
                self.assertTrue(runtime._lifecycle_lock.locked())
                mocks.register.assert_not_called()

            for initializer in mocks.configure.values():
                initializer.side_effect = initialize
            runtime.configure_observability(ObservabilityConfig())
            self.assertEqual(
                [entry[0] for entry in mocks.calls.mock_calls],
                [f"configure_{name}" for name in COMPONENTS],
            )

    def test_disabled_components_allow_startup_and_an_absent_sink_is_idempotent(self):
        with _isolated_runtime() as mocks:
            config = ObservabilityConfig()
            self.assertFalse(config.otel.enabled)
            self.assertFalse(config.langfuse.enabled)
            mocks.configure["langfuse"].return_value = None
            self.assertIsNone(runtime.configure_observability(config))
            self.assertIsNone(runtime.configure_observability(replace(config)))
            self.assertIs(runtime.running, True)
            for initializer in mocks.configure.values():
                initializer.assert_called_once()
            mocks.report.assert_not_called()
            self.assert_no_cleanup(mocks)

    def test_mutating_the_callers_nested_config_does_not_mutate_the_active_snapshot(self):
        with _isolated_runtime() as mocks:
            headers = {"x-test-header": "original"}
            config = ObservabilityConfig(otel=OTelConfig(exporter_headers=headers))
            sink = runtime.configure_observability(config)
            headers["x-test-header"] = "changed"
            self.assertEqual(runtime._active_config.otel.exporter_headers, {"x-test-header": "original"})
            previous_calls = list(mocks.calls.mock_calls)
            with self.assertRaises((RuntimeError, ValueError)):
                runtime.configure_observability(config)
            self.assertEqual(mocks.calls.mock_calls, previous_calls)
            self.assertIs(runtime._active_sink, sink)
            self.assertIs(runtime.running, True)
            self.assert_no_cleanup(mocks)

    def test_conflicts_leave_the_healthy_runtime_untouched(self):
        config = ObservabilityConfig()
        conflicts = (
            replace(config, service_name="different-service"),
            replace(config, logging=replace(config.logging, level="DEBUG")),
            replace(config, otel=replace(config.otel, enabled=True)),
            replace(config, langfuse=replace(config.langfuse, enabled=True)),
        )
        with _isolated_runtime() as mocks:
            existing = runtime.configure_observability(config)
            previous_calls = list(mocks.calls.mock_calls)
            for other_config in conflicts:
                with self.subTest(config=other_config):
                    with self.assertRaises((RuntimeError, ValueError)):
                        runtime.configure_observability(other_config)
                    self.assertIs(runtime.running, True)
                    self.assertEqual(runtime._active_config, config)
                    self.assertIs(runtime._active_sink, existing)
                    self.assertEqual(mocks.calls.mock_calls, previous_calls)
                    self.assertIs(runtime.configure_observability(config), existing)
            mocks.register.assert_called_once_with(runtime.shutdown_observability)
            self.assert_no_cleanup(mocks)

    def test_failure_at_each_initializer_cleans_owned_components_and_refuses_retry(self):
        for failed_index, component in enumerate(COMPONENTS):
            with self.subTest(component=component), _isolated_runtime() as mocks:
                failure = RuntimeError(f"startup failed: {component}")
                mocks.configure[component].side_effect = failure
                mocks.logger.hasHandlers.return_value = component != "logging"

                def check_stopped(*args, **kwargs):
                    self.assertIs(runtime.running, False)

                mocks.report.side_effect = check_stopped
                for shutdown in mocks.shutdown.values():
                    shutdown.side_effect = check_stopped
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(RuntimeError) as caught:
                        runtime.configure_observability(ObservabilityConfig())
                self.assertIs(caught.exception, failure)
                self.assertIs(runtime.running, False)
                for index, name in enumerate(COMPONENTS):
                    self.assertEqual(mocks.configure[name].call_count, int(index <= failed_index))
                self.assert_cleanup_once(mocks)
                mocks.register.assert_not_called()
                if component == "logging":
                    mocks.report.assert_not_called()
                    self.assertIn(str(failure), stderr.getvalue())
                else:
                    mocks.report.assert_called()
                    names = [entry[0] for entry in mocks.calls.mock_calls]
                    self.assertLess(names.index("report"), names.index("shutdown_langfuse"))
                previous_calls = list(mocks.calls.mock_calls)
                with self.assertRaises(RuntimeError):
                    runtime.configure_observability(ObservabilityConfig())
                runtime.shutdown_observability()
                self.assertEqual(mocks.calls.mock_calls, previous_calls)

    def test_startup_error_reporting_uses_logging_only_when_available(self):
        for mode in ("available", "no_handlers", "error_disabled", "logger_raises", "report_raises"):
            with self.subTest(mode=mode), _isolated_runtime() as mocks:
                failure = OSError("offline logging initialization failure")
                mocks.configure["logging"].side_effect = failure
                mocks.logger.hasHandlers.return_value = mode != "no_handlers"
                mocks.logger.isEnabledFor.return_value = mode != "error_disabled"
                if mode == "logger_raises":
                    mocks.logger.hasHandlers.side_effect = RuntimeError("logger unavailable")
                if mode == "report_raises":
                    mocks.report.side_effect = RuntimeError("logging failed")
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(OSError) as caught:
                        runtime.configure_observability(ObservabilityConfig())
                self.assertIs(caught.exception, failure)
                if mode in ("available", "report_raises"):
                    mocks.report.assert_called_once_with("observability_initialization_failed", failure)
                else:
                    mocks.report.assert_not_called()
                if mode == "available":
                    self.assertEqual(stderr.getvalue(), "")
                else:
                    self.assertIn(str(failure), stderr.getvalue())
                if mode not in ("no_handlers", "logger_raises"):
                    mocks.logger.isEnabledFor.assert_called_once_with(logging.ERROR)
                self.assertIs(runtime.running, False)
                mocks.register.assert_not_called()
                self.assert_cleanup_once(mocks)

    def test_cleanup_errors_never_replace_the_original_startup_exception(self):
        for reporting_fails in (False, True):
            with self.subTest(reporting_fails=reporting_fails), _isolated_runtime() as mocks:
                failure = ValueError("original startup failure")
                mocks.configure["langfuse"].side_effect = failure
                for name, shutdown in mocks.shutdown.items():
                    shutdown.side_effect = RuntimeError(f"cleanup failed: {name}")
                if reporting_fails:
                    mocks.report.side_effect = OSError("logger unavailable")
                with redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(ValueError) as caught:
                        runtime.configure_observability(ObservabilityConfig())
                self.assertIs(caught.exception, failure)
                self.assert_cleanup_once(mocks)
                reported = stderr.getvalue() + "\n".join(
                    str(invocation.args[1]) for invocation in mocks.report.call_args_list
                )
                for name in CLEANUP_ORDER:
                    self.assertIn(f"cleanup failed: {name}", reported)
                self.assertIs(runtime.running, False)

    def test_shutdown_before_startup_is_a_noop_and_does_not_prevent_startup(self):
        with _isolated_runtime() as mocks:
            runtime.shutdown_observability()
            runtime.shutdown_observability()
            self.assertIsNone(runtime.running)
            self.assertEqual(mocks.calls.mock_calls, [])
            runtime.configure_observability(ObservabilityConfig())
            self.assertIs(runtime.running, True)

    def test_shutdown_marks_stopped_before_cleanup_and_is_terminal_and_repeatable(self):
        with _isolated_runtime() as mocks:
            config = ObservabilityConfig()
            runtime.configure_observability(config)
            caller = current_thread()

            def check_stopped():
                self.assertIs(runtime.running, False)
                self.assertTrue(runtime._lifecycle_lock.locked())
                self.assertIs(current_thread(), caller)

            for shutdown in mocks.shutdown.values():
                shutdown.side_effect = check_stopped
            runtime.shutdown_observability()
            runtime.shutdown_observability()
            with self.assertRaises(RuntimeError):
                runtime.configure_observability(config)
            self.assert_cleanup_once(mocks)
            mocks.report.assert_not_called()

    def test_shutdown_continues_after_each_cleanup_failure_even_without_logging(self):
        for failed_component in CLEANUP_ORDER:
            with self.subTest(component=failed_component), _isolated_runtime() as mocks:
                runtime.configure_observability(ObservabilityConfig())
                failure = RuntimeError(f"shutdown failed: {failed_component}")
                mocks.shutdown[failed_component].side_effect = failure
                mocks.report.side_effect = OSError("logger unavailable")
                with redirect_stderr(io.StringIO()) as stderr:
                    runtime.shutdown_observability()
                    runtime.shutdown_observability()
                self.assertIn(str(failure), stderr.getvalue())
                self.assertIs(runtime.running, False)
                self.assert_cleanup_once(mocks)

    def test_concurrent_startup_or_shutdown_waits_for_full_initialization(self):
        for second_action in ("configure", "shutdown"):
            with self.subTest(second_action=second_action), _isolated_runtime() as mocks:
                entered, release = Event(), Event()
                lock = _ObservedLock()
                config = ObservabilityConfig()
                sink = mocks.configure["langfuse"].return_value

                def initialize(*args, **kwargs):
                    entered.set()
                    if not release.wait(WAIT_SECONDS):
                        raise TimeoutError("Test did not release initializer")
                    return sink

                mocks.configure["langfuse"].side_effect = initialize
                with patch.object(runtime, "_lifecycle_lock", lock), ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(runtime.configure_observability, config)
                    try:
                        self.assertTrue(entered.wait(WAIT_SECONDS))
                        second = pool.submit(
                            runtime.configure_observability, replace(config),
                        ) if second_action == "configure" else pool.submit(runtime.shutdown_observability)
                        self.assertTrue(lock.contended.wait(WAIT_SECONDS))
                        self.assertFalse(second.done())
                        self.assert_no_cleanup(mocks)
                    finally:
                        release.set()
                    self.assertIs(first.result(WAIT_SECONDS), sink)
                    result = second.result(WAIT_SECONDS)
                for initializer in mocks.configure.values():
                    initializer.assert_called_once()
                if second_action == "configure":
                    self.assertIs(result, sink)
                    self.assertIs(runtime.running, True)
                    self.assert_no_cleanup(mocks)
                else:
                    self.assertIs(runtime.running, False)
                    self.assert_cleanup_once(mocks)

    def test_concurrent_shutdown_or_restart_waits_for_full_cleanup(self):
        for second_action in ("configure", "shutdown"):
            with self.subTest(second_action=second_action), _isolated_runtime() as mocks:
                config = ObservabilityConfig()
                runtime.configure_observability(config)
                entered, release = Event(), Event()
                lock = _ObservedLock()

                def shutdown():
                    entered.set()
                    if not release.wait(WAIT_SECONDS):
                        raise TimeoutError("Test did not release cleanup")

                mocks.shutdown["langfuse"].side_effect = shutdown
                with patch.object(runtime, "_lifecycle_lock", lock), ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(runtime.shutdown_observability)
                    try:
                        self.assertTrue(entered.wait(WAIT_SECONDS))
                        self.assertIs(runtime.running, False)
                        second = pool.submit(
                            runtime.configure_observability, config,
                        ) if second_action == "configure" else pool.submit(runtime.shutdown_observability)
                        self.assertTrue(lock.contended.wait(WAIT_SECONDS))
                        self.assertFalse(second.done())
                        mocks.shutdown["otel"].assert_not_called()
                    finally:
                        release.set()
                    first.result(WAIT_SECONDS)
                    if second_action == "configure":
                        with self.assertRaises(RuntimeError):
                            second.result(WAIT_SECONDS)
                    else:
                        second.result(WAIT_SECONDS)
                self.assert_cleanup_once(mocks)

    def test_runtime_export_failure_and_full_queue_do_not_close_the_application(self):
        with _isolated_runtime() as mocks:
            config = ObservabilityConfig(langfuse=LangfuseConfig(
                enabled=True,
                queue=LangfuseQueueConfig(max_size=1, enqueue_timeout_ms=0),
            ))
            entered, release = Event(), Event()
            sink = MagicMock()

            def fail_export(event):
                entered.set()
                if not release.wait(WAIT_SECONDS):
                    raise TimeoutError("Test did not release exporter")
                raise ConnectionError("offline telemetry transport")

            sink.publish.side_effect = fail_export
            mocks.configure["langfuse"].side_effect = langfuse_client._configure_langfuse
            mocks.shutdown["langfuse"].side_effect = langfuse_client._shutdown_langfuse
            event = SimpleNamespace(llm_call_id="offline-call")
            with (
                patch.multiple(
                    langfuse_client,
                    _event_sink=langfuse_client._event_sink,
                    _event_queue=langfuse_client._event_queue,
                    _capture_enabled=False,
                    _flush_timeout_seconds=WAIT_SECONDS,
                ),
                patch.object(langfuse_client, "LangfuseSDKEventSink", return_value=sink),
                patch.object(langfuse_client, "record_langfuse_drop") as dropped,
                patch("app.services.observability.logging.log_error") as report,
                patch("app.services.observability.queue.logging.getLogger"),
            ):
                try:
                    self.assertIs(runtime.configure_observability(config), sink)
                    queue = langfuse_client._event_queue
                    self.assertIsInstance(queue, InMemoryObservabilityQueue)
                    mocks.configure["langfuse"].assert_called_once_with(
                        config.langfuse, service_name=config.service_name,
                    )
                    self.assertTrue(langfuse_client.emit_langfuse_event(event))
                    self.assertTrue(entered.wait(WAIT_SECONDS))
                    self.assertTrue(langfuse_client.emit_langfuse_event(event))
                    self.assertFalse(langfuse_client.emit_langfuse_event(event))
                    release.set()
                    self.assertTrue(queue.flush(WAIT_SECONDS))
                    self.assertEqual(report.call_count, 2)
                    dropped.assert_any_call("queue_full")
                    self.assertEqual(dropped.call_args_list.count(call("export_failed")), 2)
                    self.assertIs(runtime.running, True)
                    self.assert_no_cleanup(mocks)
                finally:
                    release.set()
                    runtime.shutdown_observability()
                self.assert_cleanup_once(mocks)
                sink.shutdown.assert_called_once_with()
                self.assertFalse(queue._worker.is_alive())


class EntryPointLifecycleTests(unittest.TestCase):
    def test_entry_points_close_after_normal_work_or_business_failure(self):
        for module_name in ("cli", "main"):
            for business_fails in (False, True):
                with self.subTest(module=module_name, business_fails=business_fails):
                    module = importlib.import_module(module_name)
                    config = _app_config()
                    lifecycle = MagicMock()
                    failure = RuntimeError("offline business failure")
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(module, "load_config", return_value=config))
                        configure = stack.enter_context(patch.object(module, "configure_observability"))
                        shutdown = stack.enter_context(patch.object(module, "shutdown_observability"))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("No network")))
                        if module_name == "cli":
                            work = stack.enter_context(patch.object(module, "_run_cli"))
                        else:
                            stack.enter_context(patch.object(module, "PydanticAIAgentRunner"))
                            orchestrator = stack.enter_context(patch.object(module, "AIOrchestrator"))
                            work = orchestrator.return_value.generate_summary
                            work.return_value = SimpleNamespace(
                                summary=SimpleNamespace(overall_summary="offline summary"),
                            )
                        lifecycle.attach_mock(configure, "configure")
                        lifecycle.attach_mock(work, "work")
                        lifecycle.attach_mock(shutdown, "shutdown")
                        if business_fails:
                            work.side_effect = failure
                            with self.assertRaises(RuntimeError) as caught:
                                module.main()
                            self.assertIs(caught.exception, failure)
                        else:
                            module.main()
                        self.assertEqual([entry[0] for entry in lifecycle.mock_calls], ["configure", "work", "shutdown"])
                        configure.assert_called_once_with(config.observability)
                        shutdown.assert_called_once_with()

    def test_startup_failure_exits_nonzero_without_business_dispatch_in_subprocess(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(BACKEND_ROOT), str(BACKEND_ROOT / ".deps")))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for module_name in ("cli", "main"):
            for component in COMPONENTS:
                with self.subTest(module=module_name, component=component), TemporaryDirectory() as cwd:
                    result = subprocess.run(
                        [sys.executable, "-c", _ENTRYPOINT_FAILURE_SCRIPT, module_name, component],
                        cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
                    )
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(f"RuntimeError: offline startup failure: {component}", result.stderr)
                    self.assertNotIn("FORBIDDEN:", output)
                    self.assertEqual(result.stdout.count("ENTRYPOINT_FINALLY"), 1, output)
                    self.assertEqual(
                        [line for line in result.stdout.splitlines() if line.startswith("CLEANUP:")],
                        [f"CLEANUP:{name}" for name in CLEANUP_ORDER], output,
                    )


_ENTRYPOINT_FAILURE_SCRIPT = r'''
from contextlib import ExitStack
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

from app.core import config
from app.services.observability import runtime

module_name, failed_component = sys.argv[1:]

def forbidden(kind):
    def fail(*args, **kwargs):
        print("FORBIDDEN:" + kind, flush=True)
        raise AssertionError(kind)
    return fail

def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        forbidden("network")()

sys.addaudithook(audit)
original_shutdown = runtime.shutdown_observability

def close():
    print("ENTRYPOINT_FINALLY", flush=True)
    original_shutdown()

with ExitStack() as stack:
    stack.enter_context(patch.object(config, "load_config", return_value=SimpleNamespace(
        observability=config.ObservabilityConfig(),
    )))
    stack.enter_context(patch.object(config, "_read_single_config_file", side_effect=forbidden("config file")))
    stack.enter_context(patch("builtins.input", side_effect=forbidden("interactive input")))
    stack.enter_context(patch("app.services.ai.orchestrator.AIOrchestrator", side_effect=forbidden("business dispatch")))
    stack.enter_context(patch("app.services.ai.pydantic_runner.PydanticAIAgentRunner", side_effect=forbidden("model runner")))
    stack.enter_context(patch.object(runtime.atexit, "register"))
    stack.enter_context(patch.object(runtime, "shutdown_observability", side_effect=close))
    stack.enter_context(patch.object(runtime, "log_error"))
    logger = stack.enter_context(patch.object(runtime, "get_logger")).return_value
    logger.hasHandlers.return_value = False
    for name in ("logging", "otel", "langfuse"):
        initialize = stack.enter_context(patch.object(runtime, "_configure_" + name))
        if name == failed_component:
            initialize.side_effect = RuntimeError("offline startup failure: " + name)
    for name in ("langfuse", "otel"):
        stack.enter_context(patch.object(
            runtime, "_shutdown_" + name,
            side_effect=lambda name=name: print("CLEANUP:" + name, flush=True),
        ))
    runpy.run_module(module_name, run_name="__main__")
'''


if __name__ == "__main__":
    unittest.main()
