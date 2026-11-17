from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FallbackTree:
    label: str
    summary: Optional[str] = None
    children: List["FallbackTree"] = field(default_factory=list)


@dataclass(frozen=True)
class StructureGraph:
    type: str
    mermaid: str
    fallback_tree: FallbackTree

