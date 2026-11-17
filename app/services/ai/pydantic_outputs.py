from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.errors import AIOutputValidationError
from app.schemas.graph import FallbackTree, StructureGraph
from app.schemas.qa import QAResult
from app.schemas.summary import SummaryResult, SummarySection
from app.services.ai.utils.text_utils import compact


class SummarySectionOutput(BaseModel):
    heading: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=1400)
    key_points: List[str] = Field(default_factory=list, max_length=8)


class SummaryOutput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    language: str = Field(pattern="^(zh|en|es|fr)$")
    overall_summary: str = Field(min_length=1, max_length=2400)
    sections: List[SummarySectionOutput] = Field(min_length=1, max_length=20)
    key_takeaways: List[str] = Field(default_factory=list, max_length=10)

    def to_result(self) -> SummaryResult:
        return SummaryResult(
            title=compact(self.title, 160),
            language=self.language,
            overall_summary=compact(self.overall_summary, 2000),
            sections=[
                SummarySection(
                    heading=compact(section.heading, 120),
                    summary=compact(section.summary, 1200),
                    key_points=[compact(point, 180) for point in section.key_points[:8] if point.strip()],
                )
                for section in self.sections
            ],
            key_takeaways=[compact(item, 180) for item in self.key_takeaways[:10] if item.strip()],
        )


class FallbackTreeOutput(BaseModel):
    label: str = Field(min_length=1, max_length=160)
    summary: Optional[str] = Field(default=None, max_length=600)
    children: List["FallbackTreeOutput"] = Field(default_factory=list, max_length=12)

    def to_result(self) -> FallbackTree:
        return FallbackTree(
            label=compact(self.label, 120),
            summary=compact(self.summary or "", 500) or None,
            children=[child.to_result() for child in self.children[:12]],
        )


class GraphOutput(BaseModel):
    type: str = Field(pattern="^(mindmap|flowchart)$")
    mermaid: str = Field(min_length=1)
    fallback_tree: FallbackTreeOutput

    def to_result(self) -> StructureGraph:
        if not (self.mermaid.startswith("mindmap") or self.mermaid.startswith("flowchart")):
            raise AIOutputValidationError("Mermaid must start with mindmap or flowchart")
        return StructureGraph(type=self.type, mermaid=self.mermaid, fallback_tree=self.fallback_tree.to_result())


class QAOutput(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)

    def to_result(self) -> QAResult:
        return QAResult(answer=compact(self.answer, 3000))


class QAPrepareOutput(BaseModel):
    search_findings: str = Field(max_length=3000)
    answer: str = Field(max_length=4000)
    feedback: str = Field(default="", max_length=1000)
