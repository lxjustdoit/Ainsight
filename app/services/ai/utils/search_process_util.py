import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from app.services.ai.utils.text_utils import normalize_text, split_sentences


NA_VALUES = {"N/A", "n/a"}


@dataclass(frozen=True)
class SearchFindingsConfig:
    min_chars: int = 30
    similarity_threshold: float = 0.12
    max_results: int = 3
    sentence_limit: int = 3
    max_chars_per_result: int = 500
    max_total_chars: int = 2500


@dataclass(frozen=True)
class InvalidTitleRules:
    invalid_exact: tuple[str, ...] = ()
    invalid_contains: tuple[str, ...] = ()
    navigation_terms: tuple[str, ...] = ()
    min_effective_chars: int = 4
    max_navigation_term_ratio: float = 0.6


class CharNGramSimilarityScorer:
    """Language-light local similarity fallback used before embedding scoring is wired in."""

    def score(self, query: str, text: str) -> Optional[float]:
        query_grams = char_ngrams(query)
        text_grams = char_ngrams(text)
        if not query_grams or not text_grams:
            return None
        return counter_cosine_similarity(Counter(query_grams), Counter(text_grams))


@dataclass
class SearchFindingsEvaluator:
    config: SearchFindingsConfig = SearchFindingsConfig()
    title_rules: InvalidTitleRules = None
    scorer: CharNGramSimilarityScorer = None

    def __post_init__(self) -> None:
        if self.title_rules is None:
            self.title_rules = load_invalid_title_rules()
        if self.scorer is None:
            self.scorer = CharNGramSimilarityScorer()

    def is_na(self, value: str) -> bool:
        return value.strip() in NA_VALUES

    def is_valid(self, *, search_findings: str, question: str, page_title: str = "") -> bool:
        findings = search_findings.strip()
        if not findings or self.is_na(findings) or len(findings) < self.config.min_chars:
            return False

        queries = [question.strip()]
        if is_valid_page_title(page_title, self.title_rules):
            queries.append(f"{question.strip()} {page_title.strip()}")

        score = max(
            (self.scorer.score(query, findings) or 0.0 for query in queries if query.strip()),
            default=None,
        )
        if score is None:
            return passes_basic_search_rules(findings)
        return score >= self.config.similarity_threshold

    def compress(self, *, search_findings: str, question: str) -> str:
        findings = search_findings.strip()
        if not findings or self.is_na(findings):
            return findings
        groups = split_result_groups(findings)
        compressed_groups: list[str] = []
        for group in groups[: self.config.max_results]:
            compressed = compress_search_result_group(
                group,
                question=question,
                sentence_limit=self.config.sentence_limit,
                max_chars=self.config.max_chars_per_result,
            )
            if compressed:
                compressed_groups.append(compressed)
        result = "\n\n".join(compressed_groups).strip()
        return result[: self.config.max_total_chars].strip()


def tokens(value: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(value))


def char_ngrams(value: str) -> list[str]:
    normalized = re.sub(r"\s+", "", normalize_text(value))
    if not normalized:
        return []
    if len(normalized) <= 2:
        return [normalized]
    grams: list[str] = []
    for size in (2, 3):
        if len(normalized) >= size:
            grams.extend(normalized[index : index + size] for index in range(0, len(normalized) - size + 1))
    return grams


def counter_cosine_similarity(left: Counter, right: Counter) -> float:
    dot = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

#embedding向量相似度计算
def vector_cosine_similarity(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    dot = sum(left_item * right_item for left_item, right_item in zip(left, right))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


@lru_cache(maxsize=4)
def load_invalid_title_rules(rule_path: str = "invalid_title_rule.json") -> InvalidTitleRules:
    path = Path(rule_path)
    if not path.is_file():
        path = Path(__file__).resolve().parents[4] / rule_path
    if not path.is_file():
        return InvalidTitleRules()
    with open(path, "r", encoding="utf-8") as rules_file:
        raw = json.load(rules_file) or {}

    groups = _language_rule_groups(raw)
    return InvalidTitleRules(
        invalid_exact=tuple(_unique_normalized(groups, "invalid_exact")),
        invalid_contains=tuple(_unique_normalized(groups, "invalid_contains")),
        navigation_terms=tuple(_unique_normalized(groups, "navigation_terms")),
        min_effective_chars=int(raw.get("min_effective_chars", 4)),
        max_navigation_term_ratio=float(raw.get("max_navigation_term_ratio", 0.6)),
    )


def is_valid_page_title(title: str, rules: Optional[InvalidTitleRules] = None) -> bool:
    rules = rules or load_invalid_title_rules()
    normalized = normalize_text(title or "")
    if len(re.sub(r"\W+", "", normalized)) < rules.min_effective_chars:
        return False
    if _looks_like_url_or_domain(normalized):
        return False
    if normalized in rules.invalid_exact:
        return False
    if any(pattern and pattern in normalized for pattern in rules.invalid_contains):
        return False
    if _navigation_term_ratio(normalized, rules.navigation_terms) >= rules.max_navigation_term_ratio:
        return False
    return True


def split_result_groups(text: str) -> list[str]:
    chunks = re.split(r"\n\s*(?=\d+\.\s+|[-*]\s+)", text.strip())
    groups = [chunk.strip() for chunk in chunks if chunk.strip()]
    if len(groups) <= 1:
        paragraphs = [chunk.strip() for chunk in re.split(r"\n{2,}", text.strip()) if chunk.strip()]
        groups = paragraphs or groups
    return groups


def compress_search_result_group(group: str, *, question: str, sentence_limit: int, max_chars: int) -> str:
    if len(group) <= max_chars:
        return group
    sentences = split_sentences(group, min_chars=8)
    if not sentences:
        return group[:max_chars].strip()
    scored = sorted(
        ((sentence, bm25_like_score(question, sentence)) for sentence in sentences),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = [sentence for sentence, _ in scored[: max(1, sentence_limit)]]
    selected = restore_original_order(sentences, selected)
    compressed = " ".join(selected).strip() or group[:max_chars].strip()
    return compressed[:max_chars].strip()


def bm25_like_score(query: str, sentence: str) -> float:
    query_terms = tokens(query)
    sentence_terms = tokens(sentence)
    if not query_terms or not sentence_terms:
        return 0.0
    counts = Counter(sentence_terms)
    length_norm = 1.0 + len(sentence_terms) / 80.0
    return sum(counts.get(term, 0) for term in set(query_terms)) / length_norm


def restore_original_order(sentences: Iterable[str], selected: list[str]) -> list[str]:
    selected_set = set(selected)
    return [sentence for sentence in sentences if sentence in selected_set]


def passes_basic_search_rules(text: str) -> bool:
    return bool(
        re.search(r"https?://|www\.|[\w-]+\.(com|org|net|io|ai|gov|edu)\b", text, re.IGNORECASE)
        or re.search(r"(^|\n)\s*(\d+\.|-|\*)\s+", text)
        or re.search(r"(source|title|url|来源|标题|摘要|事实)", text, re.IGNORECASE)
    )


def _language_rule_groups(raw: dict) -> list[dict]:
    if any(key in raw for key in ("invalid_exact", "invalid_contains", "navigation_terms")):
        return [raw]
    groups = []
    for language in ("common", "zh", "en", "es", "fr"):
        value = raw.get(language, {})
        if isinstance(value, dict):
            groups.append(value)
    return groups


def _unique_normalized(groups: list[dict], key: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for group in groups:
        for item in group.get(key, []):
            normalized = normalize_text(str(item))
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
    return values


def _looks_like_url_or_domain(value: str) -> bool:
    return bool(
        re.fullmatch(r"https?://\S+", value)
        or re.fullmatch(r"(www\.)?[\w-]+(\.[\w-]+)+(/\S*)?", value)
    )


def _navigation_term_ratio(value: str, navigation_terms: tuple[str, ...]) -> float:
    if not navigation_terms:
        return 0.0
    found_tokens = tokens(value)
    if not found_tokens:
        return 0.0
    matched = sum(1 for token in found_tokens if token in navigation_terms)
    return matched / len(found_tokens)
