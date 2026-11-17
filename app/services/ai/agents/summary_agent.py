from typing import Optional

from app.core.errors import AIOutputValidationError, InvalidSelectionError
from app.schemas.summary import SUPPORTED_SUMMARY_LANGUAGES, SummaryResult
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.prompt.prompts import build_system_prompt
from app.services.ai.pydantic_outputs import SummaryOutput
from app.services.ai.pydantic_runner import PydanticAIResult, PydanticAIRunner
from app.services.ai.utils.text_utils import split_sentences


class SummaryAgent:
    name = "SummaryAgent"

    def __init__(self, model_config: AgentModelConfig, ai_runner: PydanticAIRunner):
        self.model_config = model_config
        self.ai_runner = ai_runner

    def run(
        self,
        *,
        url: str,
        content: str,
        target_language: str,
        title: Optional[str] = None,
        selected_text: Optional[str] = None,
        summary_scope: str = "page",
    ) -> SummaryResult:
        result, _ = self.run_detailed(
            url=url,
            content=content,
            target_language=target_language,
            title=title,
            selected_text=selected_text,
            summary_scope=summary_scope,
        )
        return result

    def run_detailed(
        self,
        *,
        url: str,
        content: str,
        target_language: str,
        title: Optional[str] = None,
        selected_text: Optional[str] = None,
        summary_scope: str = "page",
    ) -> tuple[SummaryResult, PydanticAIResult]:
        return self._summarize(
            url=url,
            title=title,
            content=content,
            selected_text=selected_text,
            summary_scope=summary_scope,
            target_language=target_language,
        )

    def _summarize(
        self,
        *,
        url: str,
        title: Optional[str],
        content: str,
        selected_text: Optional[str],
        summary_scope: str,
        target_language: str,
    ) -> tuple[SummaryResult, PydanticAIResult]:
        if target_language not in SUPPORTED_SUMMARY_LANGUAGES:
            raise AIOutputValidationError(f"Unsupported summary language: {target_language}")
        if summary_scope == "selection":
            self._validate_selection(selected_text)
        elif summary_scope != "page":
            raise AIOutputValidationError(f"Unsupported summary scope: {summary_scope}")

        system_prompt = build_system_prompt(
            agent_name="Summary Assistant",
            task_rules=[
                "Generate a faithful page summary from the provided webpage text.",
                "Do not invent facts.",
                "For each original paragraph or logical section, create a short heading that captures the core idea, then summarize that section.",
                "If targetLanguage is auto, use the page's main language.",
                "Preserve important proper nouns and technical terms.",
            ],
            output_rules=[
                "The JSON object must include exactly these top-level fields: `title` string, `language` one of `zh`, `en`, `es`, `fr`, `overall_summary` string, `sections` array, and `key_takeaways` array.",
                "Each item in `sections` must include `heading` string, `summary` string, and `key_points` array of strings.",
                "Use an empty array when there are no key points or takeaways.",
                'Example output: {"title":"Page title","language":"en","overall_summary":"A concise summary.","sections":[{"heading":"Core idea","summary":"Section summary.","key_points":["Important point"]}],"key_takeaways":["Main takeaway"]}',
            ],
        )
        user_prompt = (
            f"url: {url}\n"
            f"title: {title or ''}\n"
            f"summaryScope: {summary_scope}\n"
            f"targetLanguage: {target_language}\n"
            f"selectedText: {selected_text or ''}\n"
            f"pageContent:\n{content}"
        )
        ai_result = self.ai_runner.run(
            model_config=self.model_config,
            output_type=SummaryOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return ai_result.output.to_result(), ai_result

    def _validate_selection(self, selected_text: Optional[str]) -> None:
        text = (selected_text or "").strip()
        sentence_count = len(split_sentences(text))
        if len(text) < 40 or sentence_count == 0:
            raise InvalidSelectionError(
                "Please select a more complete passage, such as a full paragraph or several continuous sentences."
            )
