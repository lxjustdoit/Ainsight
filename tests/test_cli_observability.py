"""Offline CLI lifecycle, business tracing, command, and Collector config checks."""

from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import MagicMock, call, patch

import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import cli
from app.core.config import (
    AIConfig, AIModelConfig, AppConfig, CacheConfig, EmbeddingModelConfig,
    LoggingConfig, ObservabilityConfig,
)
from app.schemas.content import PageContent
from app.schemas.graph import FallbackTree, StructureGraph
from app.schemas.qa import QAResult
from app.schemas.streaming import AIStreamEvent
from app.schemas.summary import SummaryResult
from app.services.ai.orchestrator import GraphResponse, QAResponse, SummaryResponse
from app.services.ai.pydantic_outputs import QAOutput
from app.services.ai.pydantic_runner import (
    PydanticAIResult, PydanticAITextStreamEvent, PydanticAIUsage,
)
from app.services.observability.context import get_observability_context, observability_scope
from app.services.observability.logging import _configure_logging
from app.services.observability import otel


def _config():
    model = AIModelConfig(model_name="offline-test", api_key="")
    return AppConfig(
        cache=CacheConfig(), observability=ObservabilityConfig(),
        ai=AIConfig(summary=model, graph=model, qa=model, embedding=EmbeddingModelConfig()),
    )


class CliLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch.object(cli, "load_config", return_value=self.config))
        self.configure = self.enterContext(patch.object(cli, "configure_observability"))
        self.shutdown = self.enterContext(patch.object(cli, "shutdown_observability"))
        self.runner = self.enterContext(patch.object(cli, "PydanticAIAgentRunner"))
        self.build = self.enterContext(patch.object(cli, "build_orchestrator"))
        self.enterContext(patch.object(cli, "urlopen", side_effect=AssertionError("No network")))

    def test_config_driven_initialization_and_exit_order(self):
        lifecycle = MagicMock()
        lifecycle.attach_mock(self.configure, "configure")
        lifecycle.attach_mock(self.build, "build")
        lifecycle.attach_mock(self.shutdown, "shutdown")
        with patch("builtins.input", side_effect=["", "/exit"]):
            cli.main()
        self.assertEqual(lifecycle.mock_calls, [
            call.configure(self.config.observability), call.build(self.config), call.shutdown(),
        ])
        self.assertIn("bye", self.output.getvalue())

    def test_shutdown_after_command_eof_or_interrupt(self):
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.shutdown.reset_mock()
                with patch("builtins.input", side_effect=["", error]):
                    cli.main()
                self.shutdown.assert_called_once_with()

    def test_shutdown_after_initial_prompt_interruption(self):
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.shutdown.reset_mock()
                with patch("builtins.input", side_effect=error), self.assertRaises(type(error)):
                    cli.main()
                self.shutdown.assert_called_once_with()

    def test_shutdown_after_partial_observability_initialization(self):
        self.configure.side_effect = RuntimeError("initialization failed")
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            cli.main()
        self.build.assert_not_called()
        self.shutdown.assert_called_once_with()

    def test_shutdown_after_runner_or_orchestrator_initialization_failure(self):
        for target in (self.runner, self.build):
            with self.subTest(target=target):
                self.shutdown.reset_mock()
                target.side_effect = RuntimeError("construction failed")
                with self.assertRaisesRegex(RuntimeError, "construction failed"):
                    cli.main()
                self.shutdown.assert_called_once_with()
                target.side_effect = None

    def test_direct_commands_keep_cli_context_and_continue_after_error(self):
        def check_source(*args):
            self.assertEqual(get_observability_context().request_source, "cli")
            raise RuntimeError("offline failure")

        with (
            patch.object(cli, "run_direct_qa", side_effect=check_source) as direct,
            patch.object(cli, "run_direct_qa_stream") as stream,
            patch("builtins.input", side_effect=[
                "", "/ask", "question", "plain question", "/ask-stream", "stream question",
                "/help", "/state", "/clear", "/exit",
            ]),
        ):
            cli.main()
        self.assertEqual(direct.call_count, 2)
        stream.assert_called_once_with(self.config, self.runner.return_value, "stream question")
        self.assertIn("offline failure", self.output.getvalue())
        self.shutdown.assert_called_once_with()
        self.assertIsNone(get_observability_context().request_source)

    def test_page_command_dispatch_keeps_aliases_without_recorder(self):
        def load_page(state):
            state.page = PageContent(url="https://example.test", title="Page", content="Text")

        with (
            patch.object(cli, "prompt_for_page", side_effect=load_page),
            patch.object(cli, "run_summary") as summary,
            patch.object(cli, "run_summary_stream") as summary_stream,
            patch.object(cli, "run_graph") as graph,
            patch.object(cli, "run_page_qa") as qa,
            patch.object(cli, "run_page_qa_stream") as qa_stream,
            patch("builtins.input", side_effect=[
                "/summary", "/summery", "/summary-stream", "/summery-stream", "/graph",
                "/ask", "question", "/ask-stream", "stream question", "plain question", "/exit",
            ]),
        ):
            cli.main()
        self.assertEqual(summary.call_count, 2)
        self.assertEqual(summary_stream.call_count, 2)
        graph.assert_called_once()
        self.assertEqual(qa.call_count, 2)
        qa_stream.assert_called_once()
        for command in (summary, summary_stream, graph, qa, qa_stream):
            for invocation in command.call_args_list:
                self.assertIs(invocation.args[0], self.build.return_value)
                self.assertIsInstance(invocation.args[1], cli.CliState)
                self.assertEqual(len(invocation.args), 3 if command in (qa, qa_stream) else 2)


class CliBusinessTests(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        self.output = io.StringIO()
        self.logs = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        logger = logging.getLogger("ainsight")
        handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate

        def restore_logging():
            for handler in list(logger.handlers):
                if handler not in handlers:
                    logger.removeHandler(handler)
                    handler.close()
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate

        self.addCleanup(restore_logging)
        # Preserve any pre-existing handlers when configuring this isolated test logger.
        logger.handlers = []
        _configure_logging(LoggingConfig(), stream=self.logs)
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider(shutdown_on_exit=False)
        self.addCleanup(provider.shutdown)
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.enterContext(patch.object(otel, "_otel_tracing_enabled", True))
        self.enterContext(patch.object(otel, "_tracer", provider.get_tracer("cli-tests")))
        self.enterContext(observability_scope(request_source="cli"))
        self.runner = MagicMock(spec=cli.PydanticAIAgentRunner)
        self.usage = PydanticAIUsage(input_tokens=2, output_tokens=3, total_tokens=5)

    def test_direct_qa_has_correlated_business_span_and_result_usage(self):
        self.runner.run.return_value = PydanticAIResult(QAOutput(answer="offline answer"), self.usage)
        cli.run_direct_qa(self.config, self.runner, "question")
        self.assertIn("offline answer", self.output.getvalue())
        self.assertIn("tokens=2/3/5", self.output.getvalue())
        self.assert_business_span("qa.answer", "success")

    def test_direct_stream_has_business_span_through_consumption(self):
        def events(**kwargs):
            self.assertEqual(get_observability_context().operation, "qa.stream")
            yield PydanticAITextStreamEvent(event="delta", content="offline answer")
            yield PydanticAITextStreamEvent(event="final", usage=self.usage)

        self.runner.run_text_stream.side_effect = events
        cli.run_direct_qa_stream(self.config, self.runner, "question")
        self.assertIn("offline answer", self.output.getvalue())
        self.assertEqual(self.output.getvalue().count("tokens=2/3/5"), 1)
        self.assert_business_span("qa.stream", "success")

    def test_direct_qa_failure_records_error_and_propagates(self):
        self.runner.run.side_effect = RuntimeError("offline failure")
        with self.assertRaisesRegex(RuntimeError, "offline failure"):
            cli.run_direct_qa(self.config, self.runner, "question")
        self.assert_business_span("qa.answer", "failed")

    def test_direct_stream_failure_records_error_and_propagates(self):
        def events(**kwargs):
            yield PydanticAITextStreamEvent(event="delta", content="partial")
            raise RuntimeError("stream failure")

        self.runner.run_text_stream.side_effect = events
        with self.assertRaisesRegex(RuntimeError, "stream failure"):
            cli.run_direct_qa_stream(self.config, self.runner, "question")
        self.assert_business_span("qa.stream", "failed")

    def assert_business_span(self, operation, status):
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, operation)
        self.assertEqual(span.attributes["ainsight.request_source"], "cli")
        records = [json.loads(line) for line in self.logs.getvalue().splitlines()]
        self.assertEqual(records[0]["event"], "operation_started")
        self.assertEqual(records[-1]["status"], status)
        self.assertEqual(records[-1]["event"], "business_error" if status == "failed" else "operation_finished")
        for record in records:
            self.assertEqual(record["trace_id"], f"{span.context.trace_id:032x}")
            self.assertEqual(record["span_id"], f"{span.context.span_id:016x}")
        if status == "failed":
            self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_page_results_preserve_state_and_trace_output(self):
        orchestrator = MagicMock(spec=cli.AIOrchestrator)
        state = cli.CliState(page=PageContent(url="https://example.test", title="Page", content="Text"))
        summary = SummaryResult("Title", "en", "Summary", [], [])
        graph = StructureGraph("mindmap", "mindmap\n  root", FallbackTree("root"))
        orchestrator.generate_summary.return_value = SummaryResponse("trace-summary", "ctx", "summary", summary)
        orchestrator.generate_graph.return_value = GraphResponse("trace-graph", "graph", graph)
        orchestrator.answer_question.return_value = QAResponse("trace-qa", QAResult("Answer"))
        with patch("builtins.input", return_value="en"):
            cli.run_summary(orchestrator, state)
        cli.run_graph(orchestrator, state)
        cli.run_page_qa(orchestrator, state, "question")
        self.assertEqual((state.context_id, state.summary_id, state.graph_id), ("ctx", "summary", "graph"))
        self.assertEqual(state.last_trace_id, "trace-qa")
        for trace in ("trace-summary", "trace-graph", "trace-qa"):
            self.assertIn(f"trace_id={trace}", self.output.getvalue())
        orchestrator.generate_summary.assert_called_once()

    def test_page_stream_usage_printed_once_and_state_updated(self):
        orchestrator = MagicMock(spec=cli.AIOrchestrator)
        state = cli.CliState(page=PageContent(url="https://example.test", title="Page", content="Text"))
        usage = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        orchestrator.stream_summary.return_value = [
            AIStreamEvent("final", "summary", "trace-summary", {"context_id": "ctx", "summary_id": "summary"}),
            AIStreamEvent("usage", "summary", "trace-summary", usage),
        ]
        orchestrator.qa_stream.return_value = [
            AIStreamEvent("delta", "qa", "trace-qa", {"content": "Answer"}),
            AIStreamEvent("final", "qa", "trace-qa"),
            AIStreamEvent("usage", "qa", "trace-qa", usage),
        ]
        with patch("builtins.input", return_value=""):
            cli.run_summary_stream(orchestrator, state)
        cli.run_page_qa_stream(orchestrator, state, "question")
        self.assertEqual((state.context_id, state.summary_id, state.last_trace_id), ("ctx", "summary", "trace-qa"))
        self.assertEqual(self.output.getvalue().count("tokens=2/3/5"), 2)
        orchestrator.generate_summary.assert_not_called()


class CollectorConfigurationTests(unittest.TestCase):
    def test_static_collector_signal_pipelines_and_log_mapping(self):
        config_path = Path(__file__).resolve().parents[1] / "otel-collector-config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        receivers = config["receivers"]
        self.assertEqual(receivers["otlp"]["protocols"], {
            "grpc": {"endpoint": "0.0.0.0:4317"}, "http": {"endpoint": "0.0.0.0:4318"},
        })
        filelog = receivers["filelog/ainsight"]
        self.assertEqual(filelog["include"], ["/var/log/ainsight/*.jsonl"])
        parsers = filelog["operators"]
        self.assertEqual(parsers[0]["type"], "json_parser")
        self.assertEqual(parsers[0]["parse_to"], "attributes")
        trace_parsers = [parser for parser in parsers if parser["type"] == "trace_parser"]
        self.assertEqual(len(trace_parsers), 1, "Multiple trace parsers overwrite native trace fields")
        self.assertEqual(trace_parsers[0]["trace_id"]["parse_from"], "attributes.trace_id")
        self.assertEqual(trace_parsers[0]["span_id"]["parse_from"], "attributes.span_id")
        service = next(parser for parser in parsers if parser["type"] == "copy")
        self.assertEqual(service["from"], "attributes.service_name")
        self.assertEqual(service["to"], 'resource["service.name"]')
        self.assertEqual(config["exporters"]["otlp/signoz"]["endpoint"],
                         "${env:SIGNOZ_OTLP_ENDPOINT:-host.docker.internal:4317}")
        for signal in ("traces", "metrics", "logs"):
            pipeline = config["service"]["pipelines"][signal]
            self.assertEqual(pipeline["processors"], ["memory_limiter", "batch"])
            self.assertEqual(pipeline["exporters"], ["otlp/signoz"])
            self.assertIn("otlp", pipeline["receivers"])
        self.assertIn("filelog/ainsight", config["service"]["pipelines"]["logs"]["receivers"])


if __name__ == "__main__":
    unittest.main()
