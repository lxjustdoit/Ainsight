from dataclasses import dataclass
from app.core.config import AIModelConfig


@dataclass(frozen=True)
class AgentModelConfig:
    provider: str
    model_name: str
    task_type: str
    prompt_version: str
    model_config_id: str = "local-test-model"
    base_url: str = ""
    api_key: str = ""
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


def model_config_from_app(task_type: str, config: AIModelConfig) -> AgentModelConfig:
    return AgentModelConfig(
        provider=config.provider,
        model_name=config.model_name,
        task_type=task_type,
        prompt_version=config.prompt_version,
        model_config_id=f"{config.provider}:{config.model_name}:{task_type}",
        base_url=config.base_url,
        api_key=config.api_key,
        structured_output=config.structured_output,
        tools_enabled=config.tools_enabled,
        target_output_tokens=config.target_output_tokens,
        max_output_tokens=config.max_output_tokens,
        recent_qa_messages=config.recent_qa_messages,
        temperature=config.temperature,
        input_token_price_per_1k=config.input_token_price_per_1k,
        output_token_price_per_1k=config.output_token_price_per_1k,
        search_findings_min_chars=config.search_findings_min_chars,
        search_findings_similarity_threshold=config.search_findings_similarity_threshold,
        search_findings_max_results=config.search_findings_max_results,
        search_findings_sentence_limit=config.search_findings_sentence_limit,
        search_findings_max_chars_per_result=config.search_findings_max_chars_per_result,
        search_findings_max_total_chars=config.search_findings_max_total_chars,
    )


DEFAULT_SUMMARY_MODEL = AgentModelConfig(
    provider="local",
    model_name="test-llm-summary-agent",
    task_type="summary",
    prompt_version="summary_v1",
)

DEFAULT_GRAPH_MODEL = AgentModelConfig(
    provider="local",
    model_name="test-llm-graph-agent",
    task_type="graph",
    prompt_version="graph_v1",
)

DEFAULT_QA_MODEL = AgentModelConfig(
    provider="local",
    model_name="test-llm-qa-agent",
    task_type="qa",
    prompt_version="qa_v1",
)
