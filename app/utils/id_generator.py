"""Generate shared trace and LLM call identifiers without external state."""

import secrets
import time


def _uuidv7_without_hyphen() -> str:
    """Return a UUIDv7-like sortable identifier without hyphens.

    Python 3.9 does not include uuid.uuid7, so this produces a compatible
    128-bit layout shape: 48-bit millisecond timestamp, version nibble, and
    random bits. The service prefix plus random entropy is enough for local
    uniqueness; persistence layers should still enforce uniqueness.
    """

    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (timestamp_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return f"{value:032x}"


def new_trace_id() -> str:
    # W3C trace IDs are 16-byte lowercase hexadecimal values. Keeping the
    # fallback in this format lets OTel, SigNoz, and Langfuse share one ID.
    return _uuidv7_without_hyphen()


def new_llm_call_id() -> str:
    return f"llm_{_uuidv7_without_hyphen()}"
