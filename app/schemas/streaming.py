from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class AIStreamEvent:
    event: str
    task_type: str
    trace_id: str
    data: Dict[str, Any] = field(default_factory=dict)
