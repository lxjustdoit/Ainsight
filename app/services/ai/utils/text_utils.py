import hashlib
import re
import unicodedata
from typing import List


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_text(text: str, *, lowercase: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower() if lowercase else normalized


def compact(text: str, max_chars: int) -> str:
    collapsed = normalize_text(text, lowercase=False)
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def split_sentences(text: str, *, min_chars: int = 1) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) >= min_chars]


def split_logical_sections(content: str, max_sections: int = 6) -> List[str]:
    raw_sections = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
    if not raw_sections:
        raw_sections = split_sentences(content)
    if not raw_sections:
        return []
    if len(raw_sections) <= max_sections:
        return raw_sections
    merged: List[str] = []
    bucket_size = max(1, len(raw_sections) // max_sections)
    for index in range(0, len(raw_sections), bucket_size):
        merged.append(" ".join(raw_sections[index : index + bucket_size]))
        if len(merged) == max_sections:
            tail_start = index + bucket_size
            if tail_start < len(raw_sections):
                merged[-1] = merged[-1] + " " + " ".join(raw_sections[tail_start:])
            break
    return merged


def character_windows(text: str, *, window_chars: int, overlap_chars: int = 0) -> List[str]:
    if window_chars <= 0:
        raise ValueError("window_chars must be greater than 0")
    normalized = normalize_text(text, lowercase=False)
    if not normalized:
        return []
    step = max(1, window_chars - max(0, overlap_chars))
    return [normalized[index : index + window_chars] for index in range(0, len(normalized), step)]
