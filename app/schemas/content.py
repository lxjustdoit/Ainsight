from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PageContent:
    url: str
    content: str
    title: Optional[str] = None
    selected_text: Optional[str] = None
    content_hash: Optional[str] = None

