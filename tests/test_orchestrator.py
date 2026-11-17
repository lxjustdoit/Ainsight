import io
import json
import unittest

from app.core.config import LangfuseConfig, LoggingConfig
from app.core.errors import InvalidSelectionError
from app.schemas.content import PageContent
from app.services.ai.agents.qa_agent import QAAgent
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.pydantic_runner import PydanticAIResult, PydanticAIRunner, PydanticAITextStreamEvent, PydanticAIUsage
from app.services.cache.context_cache import InMemoryContextCache
from app.services.observability.langfuse_client import InMemoryLangfuseEventSink, _configure_langfuse
from app.services.observability.logging import _configure_logging


class QueuePydanticAIRunner(PydanticAIRunner):
    def __init__(self, responses, text_responses=None, text_stream_responses=None):
        self.responses = list(responses)
        self.text_responses = list(text_responses or [])
        self.text_stream_responses = list(text_stream_responses or [])
        self.calls = []
        self.text_calls = []
        self.text_stream_calls = []

    def run(self, *, model_config, output_type, system_prompt, user_prompt, capabilities=None, retry_count=0):
        self.calls.append(
            {
                "model_config": model_config,
                "output_type": output_type,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "capabilities": capabilities or [],
                "retry_count": retry_count,
            }
        )
        if not self.responses:
            raise AssertionError("No queued Pydantic AI response")
        return PydanticAIResult(
            output=output_type.model_validate(self.responses.pop(0)),
            usage=PydanticAIUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )

    def run_text(self, *, model_config, system_prompt, user_prompt, capabilities=None, retry_count=0):
        self.text_calls.append(
            {
                "model_config": model_config,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "capabilities": capabilities or [],
                "retry_count": retry_count,
            }
        )
        if not self.text_responses:
            raise AssertionError("No queued Pydantic AI text response")
        return PydanticAIResult(
            output=self.text_responses.pop(0),
            usage=PydanticAIUsage(input_tokens=5, output_tokens=6, total_tokens=11),
        )

    def run_text_stream(self, *, model_config, system_prompt, user_prompt, capabilities=None, retry_count=0):
        self.text_stream_calls.append(
            {
                "model_config": model_config,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "capabilities": capabilities or [],
                "retry_count": retry_count,
            }
        )
        if not self.text_stream_responses:
            raise AssertionError("No queued Pydantic AI text stream response")
        text = self.text_stream_responses.pop(0)
        if isinstance(text, list):
            chunks = []
            for item in text:
                if isinstance(item, Exception):
                    raise item
                chunks.append(str(item))
                yield PydanticAITextStreamEvent(event="delta", content=str(item))
            final_text = "".join(chunks)
            yield PydanticAITextStreamEvent(
                event="final",
                content=final_text,
                usage=PydanticAIUsage(input_tokens=7, output_tokens=8, total_tokens=15),
            )
            return
        if isinstance(text, Exception):
            raise text
        midpoint = max(1, len(text) // 2)
        yield PydanticAITextStreamEvent(event="delta", content=text[:midpoint])
        yield PydanticAITextStreamEvent(event="delta", content=text[midpoint:])
        yield PydanticAITextStreamEvent(
            event="final",
            content=text,
            usage=PydanticAIUsage(input_tokens=7, output_tokens=8, total_tokens=15),
        )


def build_orchestrator(responses=None, text_responses=None, text_stream_responses=None):
    logs = io.StringIO()
    _configure_logging(LoggingConfig(), stream=logs)
    langfuse = InMemoryLangfuseEventSink()
    _configure_langfuse(LangfuseConfig(), sink=langfuse, synchronous=True)
    ai_runner = QueuePydanticAIRunner(
        responses
        or [
            {
                "title": "Readable AI",
                "language": "en",
                "overall_summary": "The page explains AI reading summaries and graph-based understanding.",
                "sections": [
                    {
                        "heading": "AI summaries clarify dense pages",
                        "summary": "AI reading tools turn dense page content into concise section summaries.",
                        "key_points": ["AI helps readers", "Summaries are section-based"],
                    }
                ],
                "key_takeaways": ["AI summaries help users understand pages"],
            },
            {
                "type": "mindmap",
                "mermaid": "mindmap\n  root((Readable AI))\n    AI summaries clarify dense pages",
                "fallback_tree": {
                    "label": "Readable AI",
                    "summary": "A structure graph for the page.",
                    "children": [{"label": "AI summaries clarify dense pages", "summary": "Summary node"}],
                },
            },
            {"answer": "The page explains how AI summaries and structure graphs help understand dense pages."},
        ],
        text_responses=text_responses,
        text_stream_responses=text_stream_responses,
    )
    orchestrator = AIOrchestrator(
        cache=InMemoryContextCache(ttl_seconds=8 * 3600),
        ai_runner=ai_runner,
    )
    return orchestrator, logs, langfuse, None, ai_runner


def log_records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


class OrchestratorTests(unittest.TestCase):
    def test_summary_graph_and_qa_have_traceable_observability(self):
        orchestrator, logs, langfuse, _, ai_runner = build_orchestrator()
        summary_response = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com",
                title="Readable AI",
                content=(
                    "AI reading tools help users understand dense pages. They produce concise section summaries.\n\n"
                    "Structure graphs help reveal relationships. Q&A should only use current page context."
                ),
            ),
        )

        graph_response = orchestrator.generate_graph(
            context_id=summary_response.context_id,
            summary_id=summary_response.summary_id,
        )
        qa_response = orchestrator.answer_question(
            context_id=summary_response.context_id,
            question="What does the page explain?",
        )

        self.assertEqual(summary_response.summary.title, "Readable AI")
        self.assertGreaterEqual(len(summary_response.summary.sections), 1)
        self.assertTrue(graph_response.graph.mermaid.startswith("mindmap"))
        self.assertEqual(qa_response.answer.based_on, "current_page_context")

        records = log_records(logs)
        self.assertEqual(len(langfuse.events), 0)
        self.assertTrue(any(record["event"] == "operation_started" for record in records))
        self.assertTrue(any(record["event"] == "operation_finished" for record in records))
        response_trace_ids = {summary_response.trace_id, graph_response.trace_id, qa_response.trace_id}
        self.assertEqual(len(response_trace_ids), 3)
        self.assertTrue(all(record["trace_id"] in response_trace_ids for record in records))
        self.assertEqual(len(ai_runner.calls), 3)
        self.assertIn("pageContent", ai_runner.calls[0]["user_prompt"])

    def test_invalid_selection_is_logged_and_recorded(self):
        orchestrator, logs, langfuse, _, ai_runner = build_orchestrator()

        with self.assertRaises(InvalidSelectionError):
            orchestrator.generate_summary(
                page=PageContent(
                    url="https://example.com",
                    title="Bad Selection",
                    content="This page has enough content to provide context for selected text.",
                    selected_text="menu",
                ),
                summary_scope="selection",
            )

        self.assertEqual(langfuse.events, [])
        self.assertEqual(len(ai_runner.calls), 0)
        error_logs = [record for record in log_records(logs) if record["event"] == "business_error"]
        self.assertEqual(error_logs[-1]["error_code"], "INVALID_SELECTION")

    def test_stream_summary_emits_items_final_and_usage(self):
        orchestrator, logs, _, _, _ = build_orchestrator(
            responses=[
                {
                    "title": "Readable AI",
                    "language": "en",
                    "overall_summary": "The page explains AI reading summaries.",
                    "sections": [
                        {
                            "heading": "AI summaries",
                            "summary": "AI summaries help users understand pages.",
                            "key_points": ["Summaries help"],
                        }
                    ],
                    "key_takeaways": ["Use summaries for scanning"],
                }
            ]
        )

        events = list(
            orchestrator.stream_summary(
                page=PageContent(
                    url="https://example.com",
                    title="Readable AI",
                    content="AI summaries help users understand pages.",
                ),
            )
        )

        self.assertEqual(events[0].event, "start")
        self.assertIn("item", [event.event for event in events])
        self.assertEqual(events[-1].event, "done")
        final = [event for event in events if event.event == "final"][0]
        self.assertEqual(final.data["summary"]["title"], "Readable AI")
        self.assertTrue(any(record["path"] == "/api/summaries/stream" for record in log_records(logs)))

    def test_stream_qa_returns_first_stage_answer_when_no_search_findings(self):
        orchestrator, _, _, _, ai_runner = build_orchestrator(
            responses=[
                {
                    "title": "Readable AI",
                    "language": "en",
                    "overall_summary": "The page explains AI reading summaries.",
                    "sections": [
                        {
                            "heading": "AI summaries",
                            "summary": "AI summaries help users understand pages.",
                            "key_points": [],
                        }
                    ],
                    "key_takeaways": [],
                },
                {
                    "search_findings": "",
                    "answer": "AI summaries help users understand dense pages.",
                    "feedback": "",
                },
            ]
        )
        summary_response = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com",
                title="Readable AI",
                content="AI summaries help users understand dense pages.",
            )
        )

        events = list(
            orchestrator.qa_stream(
                context_id=summary_response.context_id,
                question="What does the page explain?",
            )
        )

        self.assertEqual("".join(event.data.get("content", "") for event in events if event.event == "delta"), "AI summaries help users understand dense pages.")
        self.assertEqual([event.event for event in events][-1], "done")
        self.assertEqual(len(ai_runner.text_calls), 0)

    def test_qa_prepare_retry_is_reported_to_runner(self):
        orchestrator, _, _, _, ai_runner = build_orchestrator(
            responses=[
                {
                    "title": "Readable AI",
                    "language": "en",
                    "overall_summary": "The page explains AI reading summaries.",
                    "sections": [
                        {
                            "heading": "AI summaries",
                            "summary": "AI summaries help readers understand dense pages.",
                            "key_points": [],
                        }
                    ],
                    "key_takeaways": [],
                },
                {"search_findings": "N/A", "answer": "Draft answer.", "feedback": "Search failed."},
                {"search_findings": "", "answer": "Answer from current page.", "feedback": ""},
            ]
        )
        summary = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com",
                title="Readable AI",
                content="AI summaries help users understand dense pages.",
            )
        )

        list(orchestrator.qa_stream(context_id=summary.context_id, question="What does the page explain?"))

        self.assertEqual([call["retry_count"] for call in ai_runner.calls[-2:]], [0, 1])

    def test_qa_final_retry_is_reported_to_runner(self):
        runner = QueuePydanticAIRunner(
            responses=[
                {
                    "search_findings": (
                        "1. AI summaries help users understand dense pages and improve reading workflows."
                    ),
                    "answer": "Draft answer.",
                    "feedback": "",
                }
            ],
            text_responses=["", "Final concise answer."],
        )
        agent = QAAgent(
            AgentModelConfig(
                provider="local",
                model_name="test",
                task_type="qa",
                prompt_version="qa_v1",
            ),
            runner,
        )

        result, _ = agent.answer_non_stream(
            content="AI summaries help users understand dense pages.",
            question="How do AI summaries help users understand dense pages?",
            qa_messages=[],
        )

        self.assertEqual(result.answer, "Final concise answer.")
        self.assertEqual([call["retry_count"] for call in runner.text_calls], [0, 1])

    def test_stream_qa_uses_second_stage_when_search_findings_are_valid(self):
        orchestrator, _, _, _, ai_runner = build_orchestrator(
            responses=[
                {
                    "title": "Readable AI",
                    "language": "en",
                    "overall_summary": "The page explains AI reading summaries.",
                    "sections": [
                        {
                            "heading": "AI summaries",
                            "summary": "AI summaries help users understand pages.",
                            "key_points": [],
                        }
                    ],
                    "key_takeaways": [],
                },
                {
                    "search_findings": "1. Title: AI summaries\nURL: https://example.com/ai\nSummary: AI summaries help users understand dense pages and reading workflows.",
                    "answer": "Draft answer.",
                    "feedback": "",
                },
            ],
            text_stream_responses=["Final answer with search findings."],
        )
        summary_response = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com",
                title="AI summaries",
                content="AI summaries help users understand dense pages.",
            )
        )

        events = list(
            orchestrator.qa_stream(
                context_id=summary_response.context_id,
                question="How do AI summaries help?",
            )
        )

        final = [event for event in events if event.event == "final"][0]
        self.assertEqual(final.data["answer"], "Final answer with search findings.")
        self.assertEqual(len(ai_runner.text_stream_calls), 1)
        self.assertEqual(len(ai_runner.text_calls), 0)

    def test_stream_qa_returns_error_when_final_stream_fails_after_delta(self):
        orchestrator, logs, _, _, ai_runner = build_orchestrator(
            responses=[
                {
                    "title": "Readable AI",
                    "language": "en",
                    "overall_summary": "The page explains AI reading summaries.",
                    "sections": [
                        {
                            "heading": "AI summaries",
                            "summary": "AI summaries help users understand pages.",
                            "key_points": [],
                        }
                    ],
                    "key_takeaways": [],
                },
                {
                    "search_findings": "1. Title: AI summaries\nURL: https://example.com/ai\nSummary: AI summaries help users understand dense pages and reading workflows.",
                    "answer": "Draft answer.",
                    "feedback": "",
                },
            ],
            text_stream_responses=[["Partial answer", RuntimeError("stream interrupted")]],
        )
        summary_response = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com",
                title="AI summaries",
                content="AI summaries help users understand dense pages.",
            )
        )

        events = list(
            orchestrator.qa_stream(
                context_id=summary_response.context_id,
                question="How do AI summaries help?",
            )
        )

        self.assertIn("Partial answer", "".join(event.data.get("content", "") for event in events if event.event == "delta"))
        self.assertNotIn("final", [event.event for event in events])
        self.assertEqual(events[-2].event, "error")
        self.assertEqual(events[-1].event, "done")
        self.assertEqual(len(ai_runner.text_stream_calls), 1)
        finished = [record for record in log_records(logs) if record["event"] == "operation_finished"][-1]
        self.assertEqual(finished["status"], "failed")


if __name__ == "__main__":
    unittest.main()
