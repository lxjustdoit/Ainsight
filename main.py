from app.core.config import load_config
from app.schemas.content import PageContent
from app.services.ai.model_config import model_config_from_app
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.pydantic_runner import PydanticAIAgentRunner
from app.services.cache.context_cache import InMemoryContextCache
from app.services.observability.runtime import configure_observability, shutdown_observability


def main() -> None:
    config = load_config()
    try:
        configure_observability(config.observability)
        orchestrator = AIOrchestrator(
            cache=InMemoryContextCache(ttl_seconds=config.cache.page_context_ttl_hours * 3600),
            ai_runner=PydanticAIAgentRunner(),
            summary_model_config=model_config_from_app("summary", config.ai.summary),
            graph_model_config=model_config_from_app("graph", config.ai.graph),
            qa_model_config=model_config_from_app("qa", config.ai.qa),
        )
        response = orchestrator.generate_summary(
            page=PageContent(
                url="https://example.com/article",
                title="Example Article",
                content="Ainsight summarizes complex pages. It creates section summaries.\n\nIt can also create graphs and answer questions.",
            )
        )
        print(response.summary.overall_summary)
    finally:
        shutdown_observability()


if __name__ == "__main__":
    main()
