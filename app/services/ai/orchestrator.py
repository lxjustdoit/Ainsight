from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from app.core.errors import AinsightError
from app.schemas.content import PageContent
from app.schemas.graph import StructureGraph
from app.schemas.qa import QAResult
from app.schemas.streaming import AIStreamEvent
from app.schemas.summary import SummaryResult
from app.services.ai.agents.graph_agent import GraphAgent
from app.services.ai.agents.qa_agent import QAAgent
from app.services.ai.agents.summary_agent import SummaryAgent
from app.services.ai.model_config import DEFAULT_GRAPH_MODEL, DEFAULT_QA_MODEL, DEFAULT_SUMMARY_MODEL, AgentModelConfig
from app.services.ai.pydantic_runner import PydanticAIRunner, PydanticAIUsage
from app.services.ai.utils.text_utils import content_hash
from app.services.cache.context_cache import InMemoryContextCache
from app.services.observability.context import ensure_trace_id, update_observability_context
from app.services.observability.log_tracer import log_tracer
from app.services.observability.logging import log_error, log_info
from app.services.observability.otel import record_span_error, record_span_event


@dataclass(frozen=True)
class SummaryResponse:
    trace_id: str
    context_id: str
    summary_id: str
    summary: SummaryResult


@dataclass(frozen=True)
class GraphResponse:
    trace_id: str
    graph_id: str
    graph: StructureGraph


@dataclass(frozen=True)
class QAResponse:
    trace_id: str
    answer: QAResult


STREAM_CHUNK_SIZE = 36


class AIOrchestrator:
    def __init__(
        self,
        *,
        cache: InMemoryContextCache,
        ai_runner: PydanticAIRunner,
        summary_model_config: AgentModelConfig = DEFAULT_SUMMARY_MODEL,
        graph_model_config: AgentModelConfig = DEFAULT_GRAPH_MODEL,
        qa_model_config: AgentModelConfig = DEFAULT_QA_MODEL,
    ):
        self.cache = cache
        self.summary_agent = SummaryAgent(summary_model_config, ai_runner)
        self.graph_agent = GraphAgent(graph_model_config, ai_runner)
        self.qa_agent = QAAgent(qa_model_config, ai_runner)

    @log_tracer("summary.generate", task_type="summary", path="/api/summaries")
    def generate_summary(
        self,
        *,
        page: PageContent,
        target_language: str = "auto",
        summary_scope: str = "page",
        request_source: str = "test",
    ) -> SummaryResponse:
        active_trace_id = ensure_trace_id()
        resolved_hash = page.content_hash or content_hash(page.content)
        context_id = f"ctx:{resolved_hash}"
        self.cache.set(context_id, {"page": page, "qa_messages": []})
        update_observability_context(ctx=context_id)
        summary = self.summary_agent.run(
            url=page.url,
            title=page.title,
            content=page.content,
            selected_text=page.selected_text,
            summary_scope=summary_scope,
            target_language=target_language,
        )
        summary_id = f"summary:{resolved_hash}:{summary_scope}:{summary.language}"
        self.cache.set(summary_id, summary)
        update_observability_context(summary_id=summary_id)
        self._record_cache_state("summary_cached", ctx=context_id, summary_id=summary_id)
        return SummaryResponse(trace_id=active_trace_id, context_id=context_id, summary_id=summary_id, summary=summary)

    @log_tracer("summary.stream", task_type="summary", path="/api/summaries/stream")
    def stream_summary(
        self,
        *,
        page: PageContent,
        target_language: str = "auto",
        summary_scope: str = "page",
        request_source: str = "test",
    ) -> Iterable[AIStreamEvent]:
        active_trace_id = ensure_trace_id()
        yield AIStreamEvent(event="start", task_type="summary", trace_id=active_trace_id)
        try:
            resolved_hash = page.content_hash or content_hash(page.content)
            context_id = f"ctx:{resolved_hash}"
            self.cache.set(context_id, {"page": page, "qa_messages": []})
            update_observability_context(ctx=context_id)
            summary, ai_result = self.summary_agent.run_detailed(
                url=page.url,
                title=page.title,
                content=page.content,
                selected_text=page.selected_text,
                summary_scope=summary_scope,
                target_language=target_language,
            )
            summary_id = f"summary:{resolved_hash}:{summary_scope}:{summary.language}"
            self.cache.set(summary_id, summary)
            update_observability_context(summary_id=summary_id)
            self._record_cache_state("summary_cached", ctx=context_id, summary_id=summary_id)
            yield from _summary_events(active_trace_id, summary)
            yield AIStreamEvent(
                event="final",
                task_type="summary",
                trace_id=active_trace_id,
                data={
                    "context_id": context_id,
                    "summary_id": summary_id,
                    "summary": asdict(summary),
                },
            )
            yield AIStreamEvent(
                event="usage",
                task_type="summary",
                trace_id=active_trace_id,
                data=_usage_data(ai_result.usage),
            )
            yield AIStreamEvent(event="done", task_type="summary", trace_id=active_trace_id)
        except Exception as exc:
            self._record_stream_error("summary.stream", exc)
            yield AIStreamEvent(
                event="error",
                task_type="summary",
                trace_id=active_trace_id,
                data={"message": _user_error_message(exc, "当前内容暂时无法生成可靠总结，请稍后重试或选择更完整的页面内容。")},
            )
            yield AIStreamEvent(event="done", task_type="summary", trace_id=active_trace_id)

    @log_tracer("graph.generate", task_type="graph", path="/api/graphs")
    def generate_graph(
        self,
        *,
        context_id: str,
        summary_id: str,
        graph_type: str = "auto",
        request_source: str = "test",
    ) -> GraphResponse:
        active_trace_id = ensure_trace_id()
        context: Dict[str, object] = self.cache.require(context_id)
        summary: SummaryResult = self.cache.require(summary_id)
        page: PageContent = context["page"]
        graph = self.graph_agent.run(content=page.content, summary=summary, graph_type=graph_type)
        graph_id = f"graph:{summary_id}"
        self.cache.set(graph_id, graph)
        update_observability_context(graph_id=graph_id)
        self._record_cache_state("graph_cached", ctx=context_id, graph_id=graph_id)
        return GraphResponse(trace_id=active_trace_id, graph_id=graph_id, graph=graph)

    @log_tracer("qa.answer", task_type="qa", path="/api/qa")
    def answer_question(
        self,
        *,
        context_id: str,
        question: str,
        request_source: str = "test",
    ) -> QAResponse:
        active_trace_id = ensure_trace_id()
        context: Dict[str, object] = self.cache.require(context_id)
        page: PageContent = context["page"]
        qa_messages: List[str] = context.setdefault("qa_messages", [])
        answer = self.qa_agent.run(content=page.content, question=question, qa_messages=qa_messages)
        qa_messages.extend([f"user:{question}", f"assistant:{answer.answer}"])
        self.cache.set(context_id, context)
        self._record_cache_state("qa_context_updated", ctx=context_id, message_count=len(qa_messages))
        return QAResponse(trace_id=active_trace_id, answer=answer)

    @log_tracer("qa.stream", task_type="qa", path="/api/qa/stream")
    def qa_stream(
        self,
        *,
        context_id: str,
        question: str,
        request_source: str = "test",
    ) -> Iterable[AIStreamEvent]:
        active_trace_id = ensure_trace_id()
        yield AIStreamEvent(event="start", task_type="qa", trace_id=active_trace_id)
        try:
            context: Dict[str, object] = self.cache.require(context_id)
            page: PageContent = context["page"]
            qa_messages: List[str] = context.setdefault("qa_messages", [])
            plan = self.qa_agent.prepare_stream_plan(
                content=page.content,
                question=question,
                qa_messages=qa_messages,
                title=page.title or "",
            )
            answer_text = plan.answer
            usage = plan.usage
            partial_started = False
            if plan.needs_final_stream:
                try:
                    final_usage = PydanticAIUsage()
                    for stream_event in self.qa_agent.final_stream_answer(
                        content=page.content,
                        question=question,
                        search_findings=plan.search_findings,
                        draft_answer=plan.answer,
                    ):
                        if stream_event.event == "delta":
                            partial_started = True
                            yield AIStreamEvent(
                                event="delta",
                                task_type="qa",
                                trace_id=active_trace_id,
                                data={"content": stream_event.content},
                            )
                        elif stream_event.event == "final":
                            answer_text = stream_event.content
                            final_usage = stream_event.usage
                    usage = _add_usage(plan.usage, final_usage)
                except Exception:
                    if partial_started:
                        raise
                    fallback, usage = self.qa_agent.answer_non_stream(
                        content=page.content,
                        question=question,
                        qa_messages=qa_messages,
                        title=page.title or "",
                    )
                    answer_text = fallback.answer
                    for chunk in _text_chunks(answer_text):
                        yield AIStreamEvent(event="delta", task_type="qa", trace_id=active_trace_id, data={"content": chunk})
            else:
                for chunk in _text_chunks(answer_text):
                    yield AIStreamEvent(event="delta", task_type="qa", trace_id=active_trace_id, data={"content": chunk})
            yield AIStreamEvent(
                event="final",
                task_type="qa",
                trace_id=active_trace_id,
                data={"answer": answer_text, "based_on": "current_page_context"},
            )
            yield AIStreamEvent(event="usage", task_type="qa", trace_id=active_trace_id, data=_usage_data(usage))
            yield AIStreamEvent(event="done", task_type="qa", trace_id=active_trace_id)
            qa_messages.extend([f"user:{question}", f"assistant:{answer_text}"])
            self.cache.set(context_id, context)
            self._record_cache_state("qa_context_updated", ctx=context_id, message_count=len(qa_messages))
        except Exception as exc:
            self._record_stream_error("qa.stream", exc)
            yield AIStreamEvent(
                event="error",
                task_type="qa",
                trace_id=active_trace_id,
                data={"message": _user_error_message(exc, "当前问题暂时无法生成可靠回答，请稍后重试。")},
            )
            yield AIStreamEvent(event="done", task_type="qa", trace_id=active_trace_id)

    def _record_stream_error(self, operation: str, exc: Exception) -> None:
        error_code = exc.error_code if isinstance(exc, AinsightError) else "INTERNAL_ERROR"
        fields = {
            "operation": operation,
            "error_code": error_code,
            "status_code": 400 if isinstance(exc, AinsightError) else 500,
            "error_location": f"AIOrchestrator.{operation}",
        }
        record_span_error("business_error", exc, **fields)
        log_error("business_error", exc, **fields)

    def _record_cache_state(self, state: str, **fields: object) -> None:
        log_info("state_changed", state=state, **fields)
        record_span_event("state_changed", state=state, **fields)


def _summary_events(trace_id: str, summary: SummaryResult) -> Iterable[AIStreamEvent]:
    yield AIStreamEvent(event="item", task_type="summary", trace_id=trace_id, data={"type": "title", "text": summary.title})
    yield AIStreamEvent(
        event="item",
        task_type="summary",
        trace_id=trace_id,
        data={"type": "overall_summary", "text": summary.overall_summary},
    )
    for index, section in enumerate(summary.sections):
        yield AIStreamEvent(
            event="item",
            task_type="summary",
            trace_id=trace_id,
            data={
                "type": "section",
                "index": index,
                "heading": section.heading,
                "summary": section.summary,
                "key_points": section.key_points,
            },
        )
    for index, takeaway in enumerate(summary.key_takeaways):
        yield AIStreamEvent(
            event="item",
            task_type="summary",
            trace_id=trace_id,
            data={"type": "key_takeaway", "index": index, "text": takeaway},
        )


def _text_chunks(text: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterable[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def _usage_data(usage: object) -> Dict[str, int]:
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _add_usage(first: object, second: object) -> PydanticAIUsage:
    return PydanticAIUsage(
        input_tokens=int(getattr(first, "input_tokens", 0) or 0) + int(getattr(second, "input_tokens", 0) or 0),
        output_tokens=int(getattr(first, "output_tokens", 0) or 0) + int(getattr(second, "output_tokens", 0) or 0),
        total_tokens=int(getattr(first, "total_tokens", 0) or 0) + int(getattr(second, "total_tokens", 0) or 0),
        cost=0.0,
    )


def _user_error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, AinsightError):
        return exc.message
    return fallback
