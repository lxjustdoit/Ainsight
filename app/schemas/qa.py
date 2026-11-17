from dataclasses import dataclass


@dataclass(frozen=True)
class QAResult:
    answer: str
    based_on: str = "current_page_context"

