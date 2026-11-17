from app.core.errors import AIOutputValidationError
from app.schemas.graph import StructureGraph
from app.schemas.summary import SummaryResult
from app.services.ai.model_config import AgentModelConfig
from app.services.ai.prompt.prompts import build_system_prompt
from app.services.ai.pydantic_outputs import GraphOutput
from app.services.ai.pydantic_runner import PydanticAIResult, PydanticAIRunner


class GraphAgent:
    name = "GraphAgent"

    def __init__(self, model_config: AgentModelConfig, ai_runner: PydanticAIRunner):
        self.model_config = model_config
        self.ai_runner = ai_runner

    def run(self, *, content: str, summary: SummaryResult, graph_type: str = "mindmap") -> StructureGraph:
        graph, _ = self._build_graph(content=content, summary=summary, graph_type=graph_type)
        return graph

    def _build_graph(self, *, content: str, summary: SummaryResult, graph_type: str) -> tuple[StructureGraph, PydanticAIResult]:
        if not content.strip():
            raise AIOutputValidationError("Graph generation requires full page content")
        system_prompt = build_system_prompt(
            agent_name="Graph Assistant",
            task_rules=[
                "Generate a lightweight structure graph from BOTH full page content and the provided generated summary.",
                "Prefer Mermaid mindmap unless a flowchart is clearly better.",
                "Include a fallback_tree that can be rendered without Mermaid.",
            ],
            output_rules=[
                "The JSON object must include exactly these top-level fields: `type` string with value `mindmap` or `flowchart`, `mermaid` string, and `fallback_tree` object.",
                "`mermaid` must start with `mindmap` or `flowchart` and must not be wrapped in code fences.",
                "`fallback_tree` must include `label` string, `summary` string or null, and `children` array.",
                "Every child node in `children` must use the same `label`, `summary`, and `children` fields.",
                'Example output: {"type":"mindmap","mermaid":"mindmap\\n  root((Page title))\\n    Topic\\n      Detail","fallback_tree":{"label":"Page title","summary":"Brief structure summary.","children":[{"label":"Topic","summary":"Topic summary.","children":[]}]}}',
            ],
        )
        user_prompt = (
            f"requestedGraphType: {graph_type}\n"
            f"summaryTitle: {summary.title}\n"
            f"overallSummary: {summary.overall_summary}\n"
            f"summarySections: {[section.heading for section in summary.sections]}\n"
            f"fullPageContent:\n{content}"
        )
        ai_result = self.ai_runner.run(
            model_config=self.model_config,
            output_type=GraphOutput,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return ai_result.output.to_result(), ai_result
