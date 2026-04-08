from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FAQEntry:
    faq_id: str
    question_variants: list[str]
    answer: str
    tags: list[str] = field(default_factory=list)
    confirmed_by_human: bool = False
    confirmed_at: str | None = None
    freshness_hours: int | None = None
    escalate_if_stale: bool = False
    source_file: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
