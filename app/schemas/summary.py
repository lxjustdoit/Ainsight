from dataclasses import dataclass, field
from typing import List


SUPPORTED_SUMMARY_LANGUAGES = {"auto", "zh", "en", "es", "fr"}


@dataclass(frozen=True)
class SummarySection:
    heading: str
    summary: str
    key_points: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryResult:
    title: str
    language: str
    overall_summary: str
    sections: List[SummarySection]
    key_takeaways: List[str]

