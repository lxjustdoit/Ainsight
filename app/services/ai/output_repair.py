import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from pydantic import BaseModel, ValidationError

from app.core.errors import AIOutputValidationError


@dataclass(frozen=True)
class OutputValidationFeedback:
    previous_error: str
    missing_fields: List[str] = field(default_factory=list)
    invalid_fields: List[str] = field(default_factory=list)
    previous_output_preview: str = ""

    def to_prompt_text(self) -> str:
        return json.dumps(
            {
                "previous_error": self.previous_error,
                "missing_fields": self.missing_fields,
                "invalid_fields": self.invalid_fields,
                "previous_output_preview": self.previous_output_preview,
            },
            ensure_ascii=False,
        )


class OutputRepairFailure(AIOutputValidationError):
    def __init__(self, message: str, feedback: OutputValidationFeedback):
        super().__init__(message)
        self.feedback = feedback


def parse_repaired_output(raw_output: Any, *, output_type: Type[BaseModel], task_type: str) -> BaseModel:
    raw_text = _raw_to_text(raw_output)
    cleaned = _clean_model_text(raw_text)
    data = _load_json(cleaned)

    if data is None:
        if task_type == "qa" and cleaned:
            data = {"answer": cleaned}
        else:
            raise _failure("invalid_json", raw_text, invalid_fields=["json"])

    if not isinstance(data, dict):
        if task_type == "qa":
            data = {"answer": str(data)}
        else:
            raise _failure("invalid_json_object", raw_text, invalid_fields=["root"])

    data = _repair_agent_output(data, task_type)

    try:
        return output_type.model_validate(data)
    except ValidationError as exc:
        missing_fields, invalid_fields = _validation_fields(exc)
        raise _failure(
            "schema_validation_failed",
            raw_text,
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
        ) from exc


def append_regeneration_feedback(user_prompt: str, feedback: OutputValidationFeedback) -> str:
    return (
        f"{user_prompt}\n\n"
        "The previous response could not be parsed into the required JSON schema. "
        "Regenerate the answer using the same source context and fix the following structured issues. "
        "Return only one valid JSON object that matches the required schema.\n"
        f"previousFailureFeedback: {feedback.to_prompt_text()}"
    )


def _raw_to_text(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, BaseModel):
        return raw_output.model_dump_json()
    return str(raw_output)


def _clean_model_text(text: str) -> str:
    cleaned = text.replace("\ufeff", "").strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned).strip()
    cleaned = _strip_code_fences(cleaned)
    return cleaned.strip()


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.fullmatch(r"(?is)```(?:json|mermaid)?\s*(.*?)\s*```", stripped)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _load_json(text: str) -> Any:
    json_text = _extract_json_object(text) or text
    for candidate in (json_text, _repair_json_text(json_text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
    return None


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    decoder = json.JSONDecoder()
    for index in range(start, len(text)):
        if text[index] != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    return text[start:].strip()


def _repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _repair_agent_output(data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    if task_type == "qa_prepare":
        return _repair_qa_prepare(data)
    if task_type == "qa":
        return _repair_qa(data)
    if task_type == "graph":
        return _repair_graph(data)
    if task_type == "summary":
        return _repair_summary(data)
    return data


def _repair_qa(data: Dict[str, Any]) -> Dict[str, Any]:
    if "answer" not in data:
        text_values = [value for value in data.values() if isinstance(value, str) and value.strip()]
        if text_values:
            return {"answer": text_values[0].strip()}
    elif not isinstance(data["answer"], str):
        data["answer"] = str(data["answer"])
    return data


def _repair_qa_prepare(data: Dict[str, Any]) -> Dict[str, Any]:
    repaired = dict(data)
    if "search_findings" in repaired:
        repaired["search_findings"] = _normalize_prepare_text(repaired["search_findings"])
    if "answer" not in repaired and "final_answer" in repaired:
        repaired["answer"] = repaired.pop("final_answer")
    if "answer" in repaired:
        repaired["answer"] = _normalize_prepare_text(repaired["answer"])
    if "feedback" in repaired:
        repaired["feedback"] = _normalize_prepare_text(repaired["feedback"])
    else:
        repaired["feedback"] = ""
    return repaired


def _normalize_prepare_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"{index}. {item}" for index, item in enumerate(value, start=1))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _repair_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    data.setdefault("key_takeaways", [])
    return data


def _repair_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    graph_type = data.get("type")
    mermaid = _strip_code_fences(str(data.get("mermaid", ""))).strip()
    if mermaid and graph_type in {"mindmap", "flowchart"} and not mermaid.startswith(graph_type):
        mermaid = f"{graph_type}\n{mermaid}"
    data["mermaid"] = mermaid
    if isinstance(data.get("fallback_tree"), dict):
        data["fallback_tree"] = _repair_tree(data["fallback_tree"])
    return data


def _repair_tree(node: Dict[str, Any]) -> Dict[str, Any]:
    repaired = dict(node)
    repaired.setdefault("summary", None)
    repaired.setdefault("children", [])
    if isinstance(repaired["children"], list):
        repaired["children"] = [
            _repair_tree(child) if isinstance(child, dict) else {"label": str(child), "summary": None, "children": []}
            for child in repaired["children"]
        ]
    else:
        repaired["children"] = []
    return repaired


def _validation_fields(exc: ValidationError) -> tuple[List[str], List[str]]:
    missing_fields: List[str] = []
    invalid_fields: List[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", ()))
        if error.get("type") == "missing":
            missing_fields.append(loc)
        elif loc:
            invalid_fields.append(loc)
    return missing_fields, invalid_fields


def _failure(
    previous_error: str,
    raw_text: str,
    *,
    missing_fields: List[str] = None,
    invalid_fields: List[str] = None,
) -> OutputRepairFailure:
    feedback = OutputValidationFeedback(
        previous_error=previous_error,
        missing_fields=missing_fields or [],
        invalid_fields=invalid_fields or [],
        previous_output_preview=raw_text.strip()[:800],
    )
    return OutputRepairFailure(previous_error, feedback)
