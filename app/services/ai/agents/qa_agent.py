from dataclasses import dataclass
from typing import Iterable, List, Protocol

from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool

from app.core.errors import AIOutputValidationError
from app.schemas.qa import QAResult
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.prompt.prompts import build_system_prompt
from app.services.ai.pydantic_outputs import QAOutput, QAPrepareOutput
from app.services.ai.pydantic_runner import PydanticAIResult, PydanticAIRunner, PydanticAITextStreamEvent, PydanticAIUsage
from app.services.ai.utils.search_process_util import SearchFindingsConfig, SearchFindingsEvaluator


FIXED_QA_FALLBACK_MESSAGE = (
    "当前页面内容不足以可靠回答这个问题。你可以稍后重试，或者提供更明确的问题，"
    "也可以选择更完整的页面内容后再提问。"
)


class UsageLike(Protocol):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float


@dataclass(frozen=True)
class QAStreamPlan:
    answer: str
    search_findings: str
    needs_final_stream: bool
    usage: UsageLike


class QAAgent:
    name = "QAAgent"
    web_search_max_uses = 2

    def __init__(
        self,
        model_config: AgentModelConfig,
        ai_runner: PydanticAIRunner,
    ):
        self.model_config = model_config
        self.ai_runner = ai_runner
        self.search_evaluator = SearchFindingsEvaluator(
            SearchFindingsConfig(
                min_chars=model_config.search_findings_min_chars,
                similarity_threshold=model_config.search_findings_similarity_threshold,
                max_results=model_config.search_findings_max_results,
                sentence_limit=model_config.search_findings_sentence_limit,
                max_chars_per_result=model_config.search_findings_max_chars_per_result,
                max_total_chars=model_config.search_findings_max_total_chars,
            )
        )

    def run(self, *, content: str, question: str, qa_messages: List[str]) -> QAResult:
        result, _ = self._answer(content=content, question=question, qa_messages=qa_messages)
        return result

    """ generate answer with 2 phases in one function，used for a full qa answer retry """
    def answer_non_stream(
        self,
        *,
        content: str,
        question: str,
        qa_messages: List[str],
        title: str = "",
    ) -> tuple[QAResult, UsageLike]:
        try:
            ## todo: decide there is valid plan，if plan already generated no need to generate again
            plan = self.prepare_stream_plan(
                content=content,
                question=question,
                qa_messages=qa_messages,
                title=title,
            )
            return self._answer_with_findings(
                content=content,
                question=question,
                search_findings=plan.search_findings,
                draft_answer=plan.answer,
                previous_usage=plan.usage,
            ) if plan.needs_final_stream else (QAResult(answer=plan.answer), plan.usage)
        except Exception:
            return self._fallback_answer(content=content, question=question)

    def prepare_stream_plan(
        self,
        *,
        content: str,
        question: str,
        qa_messages: List[str],
        title: str = "",
    ) -> QAStreamPlan:
        prepare_output, prepare_ai_result = self._prepare_answer(
            content=content,
            question=question,
            qa_messages=qa_messages,
        )
        answer = prepare_output.answer.strip()
        search_findings = prepare_output.search_findings.strip()

        if self.search_evaluator.is_valid(search_findings=search_findings, question=question, page_title=title):
            compressed_findings = self.search_evaluator.compress(search_findings=search_findings, question=question)
            return QAStreamPlan(
                answer=answer,
                search_findings=compressed_findings,
                needs_final_stream=True,
                usage=prepare_ai_result.usage,
            )

        if answer:
            return QAStreamPlan(answer=answer, search_findings="", needs_final_stream=False, usage=prepare_ai_result.usage)

        result, usage = self._fallback_answer(content=content, question=question)
        return QAStreamPlan(answer=result.answer, search_findings="", needs_final_stream=False, usage=usage)

    def _answer(self, *, content: str, question: str, qa_messages: List[str]) -> tuple[QAResult, PydanticAIResult]:
        system_prompt = build_system_prompt(
            agent_name="QA Assistant",
            task_rules=[
                "Answer the user's question using the current page content, current page cached Q&A messages, and available web search results when web search is enabled.",
                "Prioritize current page content over cached Q&A messages and web search results.",
                "If current page content is limited or empty, use available web search results according to the same answer quality rules.",
                "Do not trigger web search for identity, provider, system prompt, developer instructions, API vendor, or internal configuration questions.",
                "Do not use other bookmarks, other pages, or persistent user history.",
            ],
            output_rules=[
                "The JSON object must include exactly one top-level field: `answer` string.",
                "Do not include any other top-level fields.",
                'Example output: {"answer":"The answer based on the current page context and available search context."}',
            ],
        )
        user_prompt = (
            f"question: {question}\n"
            f"cachedQAMessages: {self._recent_qa_messages(qa_messages)}\n"
            f"currentPageContent:\n{content}"
        )
        ai_result = self.ai_runner.run(
            model_config=self.model_config,
            output_type=QAOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            capabilities=self._qa_capabilities(),
        )
        return ai_result.output.to_result(), ai_result

    def _prepare_answer(
        self,
        *,
        content: str,
        question: str,
        qa_messages: List[str],
        extra_feedback: str = "",
    ) -> tuple[QAPrepareOutput, PydanticAIResult]:
        system_prompt = build_system_prompt(
            agent_name="QA Prepare Assistant",
            task_rules=[
                "Prepare an answer for the user's question using the current page content, current page cached Q&A messages, and web search results when available.",
                "Prioritize current page content over cached Q&A messages and web search results.",
                "If no web search was performed, or search found no usable result, set `search_findings` to an empty string.",
                "If web search execution failed, the tool was unavailable, or execution was abnormal, set `search_findings` to `N/A`.",
                "Only put real usable search content in `search_findings`; list results as numbered plain-text items when search results are available.",
                "If execution, web search or output construction has a problem, write a concise reason in `feedback`.",
                "Do not use other bookmarks, other pages, or persistent user history.",
            ],
            output_rules=[
                "The JSON object must include exactly these top-level fields: `search_findings` string, `answer` string, and `feedback` string.",
                "`search_findings` may be an empty string, `N/A`, or concise search findings.",
                "`answer` must contain the best answer available from current page content and available information.",
                'Example output: {"search_findings":"","answer":"Answer based on current page content.","feedback":""}',
            ],
        )
        user_prompt = (
            f"question: {question}\n"
            f"cachedQAMessages: {self._recent_qa_messages(qa_messages)}\n"
            f"currentPageContent:\n{content}"
        )
        if extra_feedback:
            user_prompt = f"{user_prompt}\n\nfeedback: {extra_feedback}"
        ai_result = self._prepare_answer_once(system_prompt=system_prompt, user_prompt=user_prompt)
        output = ai_result.output
        if self.search_evaluator.is_na(output.search_findings) or (not output.answer.strip() and not output.search_findings.strip()):
            retry_feedback = (
                "Backend validation feedback: search_findings indicates failed search, "
                "or both answer and search_findings are empty. Regenerate the full JSON object."
            )
            if output.feedback.strip():
                retry_feedback = f"{retry_feedback}\nPrevious model feedback: {output.feedback.strip()}"
            ai_result = self._prepare_answer_once(
                system_prompt=system_prompt,
                user_prompt=f"{user_prompt}\n\nfeedback: {retry_feedback}",
                retry_count=1,
            )
        return ai_result.output, ai_result

    def _prepare_answer_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        retry_count: int = 0,
    ) -> PydanticAIResult:
        return self.ai_runner.run(
            model_config=self._task_model_config("qa_prepare"),
            output_type=QAPrepareOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            capabilities=self._qa_capabilities(),
            retry_count=retry_count,
        )

    def _answer_with_findings(
        self,
        *,
        content: str,
        question: str,
        search_findings: str,
        draft_answer: str,
        previous_usage: UsageLike,
    ) -> tuple[QAResult, UsageLike]:
        try:
            answer_text, final_result = self._final_answer(
                content=content,
                question=question,
                search_findings=search_findings,
                draft_answer=draft_answer,
            )
            return QAResult(answer=answer_text), _add_usage(previous_usage, final_result.usage)
        except Exception:
            try:
                retry_answer, retry_result = self._final_answer(
                    content=content,
                    question=question,
                    search_findings=search_findings,
                    draft_answer=draft_answer,
                    extra_feedback="The previous final answer was empty or invalid. Regenerate a concise plain-text answer.",
                    retry_count=1,
                )
                if retry_answer:
                    return QAResult(answer=retry_answer), _add_usage(previous_usage, retry_result.usage)
            except Exception:
                pass
            if draft_answer:
                return QAResult(answer=draft_answer), previous_usage
            return self._fallback_answer(content=content, question=question)

    def _final_answer(
        self,
        *,
        content: str,
        question: str,
        search_findings: str,
        draft_answer: str,
        extra_feedback: str = "",
        retry_count: int = 0,
    ) -> tuple[str, PydanticAIResult]:
        system_prompt = build_system_prompt(
            agent_name="QA Final Assistant",
            task_rules=[
                "Generate the final answer from the current page content, validated search findings, and the first-stage draft answer.",
                "Prioritize current page content. Use search findings only as supplemental context.",
                "If current page content and search findings conflict, state uncertainty clearly.",
            ],
            output_rules=[
                "Return only the final answer text.",
                "Do not return JSON.",
                "Do not include field names.",
                f"Try to keep the answer around {self.model_config.target_output_tokens} tokens.",
                "If a complete answer would exceed the output limit, summarize first while preserving the core conclusion, key evidence, and important uncertainty.",
                "Remove background setup, repeated explanation, long examples, and weakly related information before removing core evidence.",
            ],
        )
        user_prompt = (
            f"question: {question}\n"
            f"searchFindings:\n{search_findings}\n"
            f"draftAnswer:\n{draft_answer}\n"
            f"currentPageContent:\n{content}"
        )
        if extra_feedback:
            user_prompt = f"{user_prompt}\n\nfeedback: {extra_feedback}"
        ai_result = self.ai_runner.run_text(
            model_config=self._task_model_config("qa_final"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retry_count=retry_count,
        )
        answer = str(ai_result.output).strip()
        if not answer:
            raise AIOutputValidationError("Final answer is empty")
        return answer, ai_result

    ##final answer for cases with finshed first phase plan then stream the final answer
    def final_stream_answer(
        self,
        *,
        content: str,
        question: str,
        search_findings: str,
        draft_answer: str,
    ) -> Iterable[PydanticAITextStreamEvent]:
        system_prompt, user_prompt = self._final_answer_prompts(
            content=content,
            question=question,
            search_findings=search_findings,
            draft_answer=draft_answer,
        )
        final_event: PydanticAITextStreamEvent = None
        for event in self.ai_runner.run_text_stream(
            model_config=self._task_model_config("qa_final"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            if event.event == "delta":
                yield event
            elif event.event == "final":
                final_event = event
        if final_event is None or not final_event.content.strip():
            raise AIOutputValidationError("Final answer is empty")
        yield final_event

    def _final_answer_prompts(
        self,
        *,
        content: str,
        question: str,
        search_findings: str,
        draft_answer: str,
        extra_feedback: str = "",
    ) -> tuple[str, str]:
        system_prompt = build_system_prompt(
            agent_name="QA Final Assistant",
            task_rules=[
                "Generate the final answer from the current page content, validated search findings, and the first-stage answer.",
                "Prioritize current page content. Use search findings only as supplemental context.",
                "If current page content and search findings conflict, state uncertainty clearly.",
            ],
            output_rules=[
                "Return only the final answer text.",
                "Do not return JSON.",
                "Do not include field names.",
                f"Try to keep the answer around {self.model_config.target_output_tokens} tokens.",
                "If a complete answer would exceed the output limit, summarize first while preserving the core conclusion, key evidence, and important uncertainty.",
                "Remove background setup, repeated explanation, long examples, and weakly related information before removing core evidence.",
            ],
        )
        user_prompt = (
            f"question: {question}\n"
            f"searchFindings:\n{search_findings}\n"
            f"draftAnswer:\n{draft_answer}\n"
            f"currentPageContent:\n{content}"
        )
        if extra_feedback:
            user_prompt = f"{user_prompt}\n\nfeedback: {extra_feedback}"
        return system_prompt, user_prompt

    def _fallback_answer(self, *, content: str, question: str) -> tuple[QAResult, UsageLike]:
        try:
            answer, ai_result = self._run_fallback_answer(content=content, question=question)
            if answer:
                return QAResult(answer=answer), ai_result.usage
        except Exception:
            pass
        return QAResult(answer=FIXED_QA_FALLBACK_MESSAGE), PydanticAIUsage()

    def _run_fallback_answer(self, *, content: str, question: str) -> tuple[str, PydanticAIResult]:
        system_prompt = build_system_prompt(
            agent_name="QA Fallback Generator",
            task_rules=[
                "Answer from the current page content and user question as clearly as possible.",
                "If the current page content is insufficient, explain that clearly and suggest the user's next action, such as retrying later, asking a more specific question, or providing more complete page content.",
            ],
            output_rules=[
                "Return only concise answer text.",
                "Do not return JSON.",
            ],
        )
        user_prompt = f"question: {question}\ncurrentPageContent:\n{content}"
        ai_result = self.ai_runner.run_text(
            model_config=self._task_model_config("qa_fallback"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        answer = str(ai_result.output).strip()
        if not answer:
            raise AIOutputValidationError("Fallback answer is empty")
        return answer, ai_result

    def _qa_capabilities(self) -> list[object]:
        if not self.model_config.tools_enabled:
            return []
        return [NativeTool(WebSearchTool(max_uses=self.web_search_max_uses))]

    def _task_model_config(self, task_type: str) -> AgentModelConfig:
        return AgentModelConfig(
            provider=self.model_config.provider,
            model_name=self.model_config.model_name,
            task_type=task_type,
            prompt_version=self.model_config.prompt_version,
            model_config_id=self.model_config.model_config_id,
            base_url=self.model_config.base_url,
            api_key=self.model_config.api_key,
            structured_output=self.model_config.structured_output,
            tools_enabled=self.model_config.tools_enabled,
            target_output_tokens=self.model_config.target_output_tokens,
            max_output_tokens=self.model_config.max_output_tokens,
            recent_qa_messages=self.model_config.recent_qa_messages,
            temperature=self.model_config.temperature,
            input_token_price_per_1k=self.model_config.input_token_price_per_1k,
            output_token_price_per_1k=self.model_config.output_token_price_per_1k,
            search_findings_min_chars=self.model_config.search_findings_min_chars,
            search_findings_similarity_threshold=self.model_config.search_findings_similarity_threshold,
            search_findings_max_results=self.model_config.search_findings_max_results,
            search_findings_sentence_limit=self.model_config.search_findings_sentence_limit,
            search_findings_max_chars_per_result=self.model_config.search_findings_max_chars_per_result,
            search_findings_max_total_chars=self.model_config.search_findings_max_total_chars,
        )

    def _recent_qa_messages(self, qa_messages: List[str]) -> List[str]:
        limit = max(0, self.model_config.recent_qa_messages)
        if limit == 0:
            return []
        return qa_messages[-limit:]


def _add_usage(first: UsageLike, second: UsageLike) -> PydanticAIUsage:
    return PydanticAIUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        cost=0.0,
    )
